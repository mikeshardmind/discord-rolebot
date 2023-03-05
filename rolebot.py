"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.

Copyright (C) 2022 Michael Hall <https://github.com/mikeshardmind>
"""

from __future__ import annotations

import json
import logging
import os
import re
import struct
from collections.abc import Iterator
from itertools import chain
from pathlib import Path
from typing import NamedTuple

import base2048
import discord
import xxhash
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESSIV  # See security considerations below if concerned
from typing_extensions import Self

try:
    import tomllib  # type: ignore
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore

log = logging.getLogger(__name__)

##################################### SECURITY CONSIDERATIONS  #####################################
#
# The key is kept secret
# Associated data is known by both parties, [guild_id, channel_id, message_id]
# {plaintext, associated_data} is a never-reused combination.
#
# Not in scope: Compromise of key via access to the key outside of what this service enables.
#
# Primary concern & scope: Spoofed custom_id in interaction payload from discord.
#
# The original concerns here involved that in the past it was possible for users to spoof
# interaction data and discord would not validate it before sending it to bots.
# According to a conversation in the discord.py discord server, this should no longer be possible.
# Given the potential negative impact, as well as discord's track record, it's unacceptable to
# trust that this will remain in a "safe" state.
# Recent issue:  XSS in December 2022
# Previous issues: this, various misunderstandings of sanitization and validation issues, etc
#
# Addtiional consideration: Known plaintext with AEESIV
#
# As the code here is open source, and the users effectively may know the plaintext
# Within restriction based on discord snowflake format,
# and limitations of the constrained scheme below, users may choose the plain text.
# The effort involved in a Known plaintext attack must be considered.
# In the case of such an attack, a user could attempt to attack the key.
# As we do not differentiate to an attacker any of the potential reasons this could fail,
# The attack would be limited by all of the below factors
#
# 1. Discord interaction ratelimits.
#       (TODO: add throttling on the bot end to ensure this isn't another discord failure point.)
# 2. The ability to spoof interaction data
#       (The window to attack would be limited to a time when this is possible) (See above)
# 3. Not entirely up to the user: plaintext
# 4. Not chosen by the user: associated data.
# 5. Attempts at this would raise suspicion based on logging
#       (weak until TODO: alarming on specific exceptions not yet implemented)
# 6. Existing resistance against such attacks in AES-SIV
#   - `Deterministic Authenticated-Encryption: A Provable-Security Treatment of the Key-Wrap Problem`
#   - Sections 6-8
#   - Link: https://eprint.iacr.org/2006/221
# 7. Arguments of economics as well as thermodynamic ideals
#   - See https://www.schneier.com/blog/archives/2009/09/the_doghouse_cr.html
#
# If you find a security issue in this, please reach out.
#
# POTENTIAL IMPACT:
# Successful spoofing of this data could lead to a malicious user granting
# themselves more permissions and roles in discord. Such escalation can
# Can indirectly lead to more than the permissions the roles themselves have
# in certain cases involving bots which use roles to indicate priveleged user groups
# rather than use the permissions of the roles directly
#
# No direct security impact to host, potential security impact to users of the provided service.
#
####################################################################################################


########################################## Struct layouts ##########################################
# Space available: 100 utf-8 characters. 5 charcters used for plaintext prefix matching.
# Assumptions: base2048, 95 utf-8 characters, full bytes only, aessiv with a 256 bit key
# Note: 256 bit key corresponds to AESSIV.generate_key(512)
# Effective space is 95 characters of base2048 in byte increments, or 130 bytes
# AES-SIV has a 16 byte overhead to it in these conditions, leaving 114 bytes
# We can reasonably version within a single byte
# Each discord id is 8 bytes, and we have 6 discrete sets of ids
# We also need to reserve space for a version to allow future changes as needed.
# --------------------------------------------------------------------------------------------------
# Common:
# up to 112 byte payload
# 1-byte: version | up to 111 bytes: versioned data
# --------------------------------------------------------------------------------------------------
# Version 1:
# 1-byte: version = x01 | 6 length prefixed arrays of discord ids to be used in application as a set
#                           legnth prefix as 1 byte, elements 8 bytes each representing a 64bit int
# each of these sets are used in a ruleset to represent a combination of actions and conditions:
# (in order) to add, to remove, to toggle, to require any of, to require all of, to forbid any of
####################################################################################################


class NoUserFeedback(Exception):
    """Used to mark exceptions that should consistently just mark something a failure"""


class RuleError(Exception):
    """Used to indicate issues with specific rules"""


class V1TooManyIDs(RuleError):
    def __init__(self: Self, *args: object) -> None:
        super().__init__("May only pack 13 discord ids into a version 1 ruleset")


class V1ToggleWithAddRemove(RuleError):
    def __init__(self: Self, *args: object) -> None:
        super().__init__("Can't mix toggle with add or remove")


class V1MultipleToggle(RuleError):
    def __init__(self: Self, *args: object) -> None:
        super().__init__("May only toggle 1 role at once")


class V1NonActionableRule(RuleError):
    def __init__(self: Self, *args: object) -> None:
        super().__init__("Must provide at least one action from add, remove, or toggle")


class RulesFileError(Exception):
    """Used to indicate errors involving user-provided rules files"""


class NoContent(RulesFileError):
    def __init__(self: Self, *args: object) -> None:
        super().__init__("Must provide content for the message the menu will be attached to")


class NonStringLabel(RulesFileError):
    def __init__(self: Self, *args: object) -> None:
        super().__init__("Label must be a string")


class NonStringContent(RulesFileError):
    def __init__(self: Self, *args: object) -> None:
        super().__init__("`Content` key must contain a string")


class BadIDList(RulesFileError):
    def __init__(self: Self, bad_key: str, *args: object) -> None:
        self.bad_key = bad_key
        super().__init__(f"Key: {bad_key} should be a list of integer role ids")


class UserFacingError(Exception):
    def __init__(self: Self, error_message: str, *args: object) -> None:
        self.error_message = error_message
        super().__init__(*args)


class NoSuchRole(UserFacingError):
    def __init__(self: Self, role_id: int, *args: object) -> None:
        super().__init__(f"I can't find a role with id {role_id}")


class UserHierarchyIssue(UserFacingError):
    def __init__(self: Self, role: discord.Role, *args: object) -> None:
        super().__init__(f"You can't create a rule for {role.mention} as it could violate discord role hierarchy")


class BotHierarchyIssue(UserFacingError):
    def __init__(self: Self, role: discord.Role, *args: object) -> None:
        super().__init__(f"I can't assign {role.mention} as it is not below my top role.")


class CantAssignManagedRole(UserFacingError):
    def __init__(self: Self, role: discord.Role, *args: object) -> None:
        super().__init__(f"{role.mention} is managed by an integration and cannot be assigned like this.")


class DataV1(NamedTuple):
    add: frozenset[int]
    remove: frozenset[int]
    toggle: frozenset[int]
    require_any: frozenset[int]
    require_all: frozenset[int]
    require_none: frozenset[int]

    def validate(self: Self) -> None:
        """
        Checks that
        - there are 13 or fewer discord ids encoded
        - that toggle is not provided with either add or remove
        - that toggle contains no more than 1 role
        - that at least one of toggle, add, or remove is provided
        """
        if sum(map(len, self)) > 13:
            raise V1TooManyIDs

        if tog := self.toggle:
            if self.add or self.remove:
                raise V1ToggleWithAddRemove
            if len(tog) > 1:
                raise V1MultipleToggle
        else:
            if not (self.add or self.remove):
                raise V1NonActionableRule

    def check_ids_meet_requirements(self: Self, ids: set[int], /) -> bool:
        """
        Checks if a hypothetical set of ids meet requirements
        """

        if self.require_any and not (self.require_any & ids):
            return False

        if (self.require_all & ids) - self.require_all:
            return False

        if self.require_none & ids:
            return False

        return True

    def apply_to_ids(self: Self, ids: set[int], /) -> set[int]:
        return ((ids ^ self.toggle) | self.add) - self.remove


# can expand this to a type union if needed later on,
# until then it's an alias where future API should handle this
VERSIONED_DATA = DataV1


def pack_rules(data: VERSIONED_DATA, /) -> bytes:
    data.validate()
    struct_fmt = "!bb%dQb%dQb%dQb%dQb%dQb%dQ" % tuple(map(len, data))
    to_pack = chain.from_iterable((len(lst), *lst) for lst in data)
    return struct.pack(struct_fmt, 1, *to_pack)  # needs changing if new version


def _v1_struct_unpacker(raw: bytes, /) -> Iterator[frozenset[int]]:
    """
    Calling contract is that you have checked the version in advance
    """
    offset: int = 1
    for _ in range(6):
        (_len,) = struct.unpack_from("!b", raw, offset)
        yield frozenset(struct.unpack_from("!%dQ" % _len, raw, offset + 1))
        offset += 8 * _len + 1


def _get_data_version(b: bytes, /) -> int:
    (r,) = struct.unpack_from("!b", b, 0)
    assert isinstance(r, int)
    return r


def unpack_rules(raw: bytes, /) -> VERSIONED_DATA:
    """
    Errors here should not be reported to users.
    These should be pass/fail only to them.

    Potential place to consider impact of a timing attack,
    but I doubt what's here is meaningfully susceptible to that
    when it should be eclipsed by the networking delays from
    bot <-> discord <-> user
    as well as by various ratelimits
    (already inconsistent response time for a basic always respond interaction)
    """
    try:
        version = _get_data_version(raw)
    except struct.error:
        raise NoUserFeedback from None

    if version != 1:
        raise NoUserFeedback

    try:
        data = DataV1(*_v1_struct_unpacker(raw))
        data.validate()
    except (NoUserFeedback, struct.error):
        raise NoUserFeedback from None
    except Exception as exc:
        log.exception("Unhandled exception type", exc_info=exc)
        raise NoUserFeedback from None

    return data


def get_secret_data_from_file(path: Path) -> tuple[int, AESSIV]:
    """
    We get back a number that represents an equivalent
    time portion of a discord snowflake (snowflake >> 22)
    for when a key was generated ad well as a key
    """
    with path.open(mode="rb") as fp:
        raw = fp.read()
        ts, k = struct.unpack("!Q64s", raw)
        return ts, AESSIV(k)


def _generate_secret_data_file(path: Path) -> None:
    """
    Writes a file in the format expected by `get_secret_data`
    after generating the values and writes to a path.

    assumes system clock is fine.

    will not overwrite an existing path
    (barring filesytem race conditions not considered in scope to account for)
    """

    key = AESSIV.generate_key(512)
    time = discord.utils.time_snowflake(discord.utils.utcnow()) >> 22

    try:
        with path.open(mode="xb") as fp:
            fp.write(struct.pack("!Q64s", time, key))
    except FileExistsError:
        pass


def parse_rules_file(
    raw: bytes,
) -> tuple[str | None, discord.Embed | None, list[tuple[str, VERSIONED_DATA]]]:

    # This whole thing is completely not understood by typechecking,
    # and we're gonna validate this anyhow...

    loaded = tomllib.loads(raw.decode("utf-8"))  # type: ignore
    content = loaded.get("content", None)  # type: ignore
    ret_buttons: list[tuple[str, VERSIONED_DATA]] = []

    if not isinstance(content, str):
        raise NonStringContent

    if not content.strip():  # type: ignore
        raise NoContent

    for button in loaded.get("buttons", []):  # type: ignore
        label = button.get("label", None)  # type: ignore

        # No support for embeds yet

        if not isinstance(label, str):
            raise NonStringLabel

        sets: list[frozenset[int]] = []

        for name in ("add", "remove", "toggle", "require_any", "require_all", "require_none"):
            rids = button.get(name, ())  # type: ignore
            if not isinstance(rids, (list, tuple)):
                raise BadIDList(name)
            if not all(isinstance(element, int) for element in rids):  # type: ignore
                raise BadIDList(name)
            sets.append(frozenset(rids))  # type: ignore

        rules = DataV1(*sets)
        rules.validate()

        ret_buttons.append((label, rules))

    return content, None, ret_buttons  # type: ignore # validated above


def parse_and_check_rules(
    guild: discord.Guild,
    user: discord.Member,
    raw: bytes,
) -> tuple[str | None, discord.Embed | None, list[tuple[str, VERSIONED_DATA]]]:
    generic_msg = "Something was wrong with that file (More detailed errors may be provided in the future.)"
    try:
        content, embed, labeled_rules = parse_rules_file(raw)
    except (KeyError, RuleError, TypeError):
        raise UserFacingError(generic_msg) from None

    for _label, rules in labeled_rules:
        for rid in chain(rules.add, rules.remove, rules.toggle):
            role = guild.get_role(rid)
            if role is None:
                raise NoSuchRole(rid)
            if role >= user.top_role and user.id != guild.owner_id:
                raise UserHierarchyIssue(role)
            if role >= guild.me.top_role and guild.me.id != guild.owner_id:
                raise BotHierarchyIssue(role)

    return content, embed, labeled_rules


@discord.app_commands.guild_only()
@discord.app_commands.default_permissions(manage_roles=True)
async def role_menu_maker(itx: discord.Interaction[RoleBot], attachment: discord.Attachment) -> None:
    """
    This function creates a role menu from an attachment.

    To be better documented soon :tm:
    May also create a web interface, who knows.
    """
    # guild only decorator, not checking for discord/discord.py failures here
    assert isinstance(itx.user, discord.Member)
    assert itx.guild

    if not isinstance(itx.channel, discord.TextChannel):
        # I don't think this is gonna come up??
        await itx.response.send_message("Use this in a text channel instead", ephemeral=True)
        return

    # Even if someone modifies the app command settings, we aren't gonna allow this. It's privesc
    if not itx.user.guild_permissions.manage_roles:
        await itx.response.send_message("You can't do this. (missing permissions)", ephemeral=True)
        return

    await itx.response.defer(ephemeral=True)

    try:
        content, embed, labeled_rules = parse_and_check_rules(itx.guild, itx.user, await attachment.read())
    except UserFacingError as exc:
        await itx.followup.send(exc.error_message, ephemeral=True)
        return

    try:
        await itx.client.create_role_menu(
            itx.channel,
            content=content,
            embed=embed,
            label_action_pairs=labeled_rules,
        )
    except (RuleError, RulesFileError, discord.HTTPException):
        await itx.followup.send("Something went wrong. (more detailed error in future versions)")
        return

    # It'll get stuck thinking without this, thanks discord...
    await itx.followup.send("Menu created", ephemeral=True)


class VersionableTree(discord.app_commands.CommandTree):
    async def get_hash(self: Self) -> bytes:
        commands = sorted(self._get_all_commands(guild=None), key=lambda c: c.qualified_name)

        translator = self.translator
        if translator:
            payload = [await command.get_translated_payload(translator) for command in commands]
        else:
            payload = [command.to_dict() for command in commands]

        return xxhash.xxh64_digest(json.dumps(payload).encode("utf-8"), seed=0)


class RoleBot(discord.AutoShardedClient):
    def __init__(self: Self, valid_since: int, aessiv: AESSIV) -> None:
        super().__init__(intents=discord.Intents(1))
        self.aessiv = aessiv
        self.valid_since = valid_since
        self.interaction_regex = re.compile(r"^rr\d{2}:(.*)$", flags=re.DOTALL)
        self.tree = VersionableTree(self, fallback_to_global=False)

    async def setup_hook(self: Self) -> None:
        """
        Someone will come along and say not to do it this way, and I'll ask them
        to figure out a better way without wrecking type information
        or having a command tied to an instance of a bot as module state.
        """
        self.tree.command(name="createrolemenu")(role_menu_maker)

        tree_hash = await self.tree.get_hash()

        path = Path(__file__).with_name("rolebot.syncdata")
        with path.open(mode="w+b") as fp:
            data = fp.read()
            if data != tree_hash:
                await self.tree.sync()
                fp.seek(0)
                fp.write(tree_hash)

        await self.tree.sync()

    async def on_interaction(self: Self, interaction: discord.Interaction[Self]) -> None:
        if interaction.type is discord.InteractionType.component:
            # components *should* be guaranteed to have a custom_id by discord
            assert interaction.data is not None
            custom_id = interaction.data.get("custom_id", "")
            # above avoids relying on non-publicly exported types in an assertion
            if m := self.interaction_regex.match(custom_id):
                guild = interaction.guild
                if guild is None or not guild.me.guild_permissions.manage_roles:
                    await interaction.response.defer()
                    return
                await self.handle_rules_for_interaction(interaction, m.group(1))

    def _decrypt_and_parse_rules(
        self: Self,
        msg_id: int,
        user_id: int,
        encoded_rules: str,
    ) -> VERSIONED_DATA:
        """
        Handles decryption and parsing of rules, logging / raising if needed.
        """
        if (msg_id >> 22) < self.valid_since:
            log.info(
                "Not processing button that predates key on message_id: %d from user: %d",
                msg_id,
                user_id,
            )
            raise NoUserFeedback

        try:
            by = base2048.decode(encoded_rules)
        except Exception:
            log.exception(
                "SEC: Could not decode expected valid custom id using base2048: %s from user %d",
                encoded_rules,
                user_id,
            )
            raise NoUserFeedback from None

        try:
            packed_rules = self.aessiv.decrypt(by, [msg_id.to_bytes(8, byteorder="big")])
        except InvalidTag:
            log.exception(
                "SEC: Got a custom id %s that couldn't decrypt with aessiv key from user %d",
                encoded_rules,
                user_id,
            )
            raise NoUserFeedback from None

        try:
            rules = unpack_rules(packed_rules)
        except NoUserFeedback:
            log.exception(
                "SEC: Could not unpack custom_id %s after decryption passed from user %d",
                encoded_rules,
                user_id,
            )
            raise

        return rules

    @staticmethod
    def _preempt_role_violations(
        interaction: discord.Interaction,
        rules_pack: str,
        added: set[int],
        removed: set[int],
    ) -> None:
        guild = interaction.guild
        message = interaction.message

        assert guild
        assert message

        for rid in chain(added, removed):
            if role := guild.get_role(rid):
                if role.managed:
                    # this should only happen if a role can *become* managed. not putting it past discord to change this
                    CantAssignManagedRole(role)
                if role >= guild.me.top_role:
                    raise BotHierarchyIssue(role)
            else:
                # SEC: There's a balance between security and usability to consider below
                # The ideal for security is to silently fail here too.
                # The reality is that successfully attacking to this point
                # and not tripping prior alarms for an extended period measuring years...
                # If the id is impossible, it must be part of an attempt at attacking this or a major flaw
                # in the way something is handled to get to this state.
                # Otherwise, given the various factors at play, we'll assume the benign case of a role being deleted
                # after a menu was created referencing it.

                if (rid.bit_length() < 43) or (  # impossible snowflake
                    (guild.id >> 22) < (rid >> 22) < (message.id >> 22)
                ):  # temporally anomalous
                    log.critical(
                        "SEC: passed decryption, passed unpack, impossible ID %d from packed rules %s from user %d",
                        rid,
                        rules_pack,
                        interaction.user.id,
                    )
                    raise NoUserFeedback
                raise NoSuchRole(rid)

    async def handle_rules_for_interaction(
        self: Self,
        interaction: discord.Interaction[Self],
        encoded_rules: str,
    ) -> None:
        await interaction.response.defer(ephemeral=True)

        # calling contract here ensures:
        # 1. That the message exists
        # 2. That the guild exists
        # 3. that the user is a member
        assert interaction.message is not None
        assert isinstance(interaction.user, discord.Member)
        message = interaction.message
        user = interaction.user

        rules = self._decrypt_and_parse_rules(message.id, user.id, encoded_rules)

        starting_role_ids = {r.id for r in user.roles}

        if not rules.check_ids_meet_requirements(starting_role_ids):
            await interaction.followup.send(
                content="You are ineligible to use this button",
                ephemeral=True,
            )
            return

        new_role_ids = rules.apply_to_ids(starting_role_ids)
        # not using symetric difference here because we need these seperately later anyhow
        added = new_role_ids - starting_role_ids
        removed = starting_role_ids - new_role_ids

        if not (added or removed):
            return

        try:
            self._preempt_role_violations(interaction, encoded_rules, added, removed)
        except UserFacingError as exc:
            await interaction.followup.send(content=exc.error_message, ephemeral=True)
            return
        except NoUserFeedback:
            return

        try:
            await user.edit(
                roles=[discord.Object(rid) for rid in new_role_ids],
                reason="Used a role button.",
            )
        except discord.HTTPException as exc:
            log.exception(
                "Error assigning role from button when expected to work code: %d  status: %d",
                exc.code,
                exc.status,
                exc_info=exc,
            )
        else:
            add_str = rem_str = ""
            if added:
                add_str = "Added roles: " + ", ".join(f"<@&{rid}>" for rid in added)
            if removed:
                rem_str = "Removed roles: " + ", ".join(f"<@&{rid}>" for rid in removed)

            content = "\n".join(filter(None, (add_str, rem_str)))
            await interaction.followup.send(content=content, ephemeral=True)

    async def create_role_menu(
        self: Self,
        channel: discord.TextChannel,
        content: str | None,
        embed: discord.Embed | None,
        label_action_pairs: list[tuple[str, VERSIONED_DATA]],
    ) -> discord.Message:

        # We shouldn't be able to hit this with this being checked in file parsing
        if len(label_action_pairs) > 25:
            msg = "Cannot provide that many buttons in a view"
            raise ValueError(msg)

        view = discord.ui.View(timeout=None)
        view.stop()

        message: discord.Message

        if embed and content:
            message = await channel.send(content=content, embed=embed)
        elif embed:
            message = await channel.send(embed=embed)
        elif content:
            message = await channel.send(content=content)
        else:
            msg = "Must provide an embed or content"
            # We shouldn't be able to hit this with this being checked in file parsing
            raise ValueError(msg)

        for idx, (label, data) in enumerate(label_action_pairs):
            packed = pack_rules(data)
            enc = self.aessiv.encrypt(packed, [message.id.to_bytes(8, byteorder="big")])
            to_disc = base2048.encode(enc)
            custom_id = f"rr{idx:02}:{to_disc}"
            button: discord.ui.Button[discord.ui.View] = discord.ui.Button(label=label, custom_id=custom_id)
            view.add_item(button)

        try:
            await message.edit(view=view)
        except Exception as exc:
            log.exception("edit fail", exc_info=exc)
            try:
                await message.delete()
            except discord.HTTPException:
                pass

        return message


def _get_token() -> str:
    # TODO: keyrings, systemdcreds, etc
    token = os.getenv("ROLEBOT_TOKEN")
    if not token:
        tp = Path(__file__).with_name("rolebot.token")
        try:
            with tp.open(mode="r") as fp:
                token = fp.read().strip()
        except OSError:
            msg = "NO TOKEN? (Use Environment `ROLEBOT_TOKEN` or file `rolebot.token`)"
            raise RuntimeError(msg) from None
    return token


def main() -> None:
    token = _get_token()
    p = Path(__file__).with_name("rolebot.secrets")
    _generate_secret_data_file(p)
    valid_since, aessiv = get_secret_data_from_file(p)

    client = RoleBot(valid_since, aessiv)
    client.run(token)


if __name__ == "__main__":
    main()
