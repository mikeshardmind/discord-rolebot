"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.

Copyright (C) 2022 Michael Hall <https://github.com/mikeshardmind>
"""

from __future__ import annotations

import logging
import re
import struct
from itertools import chain
from pathlib import Path
from typing import Iterator, NamedTuple

import base2048
import discord
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESSIV
from typing_extensions import Self

log = logging.getLogger(__name__)

########################################## HAZMAT: AESSIV ##########################################
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


class DataV1(NamedTuple):
    add: frozenset[int]
    remove: frozenset[int]
    toggle: frozenset[int]
    require_any: frozenset[int]
    require_all: frozenset[int]
    require_none: frozenset[int]

    def validate(self):
        """
        Checks that
        - there are 13 or fewer discord ids encoded
        - that toggle is not provided with either add or remove
        - that toggle contains no more than 1 role
        - that at least one of toggle, add, or remove is provided
        """
        if sum(map(len, self)) > 13:
            raise RuleError("May only pack 13 discord ids into a version 1 ruleset")

        if tog := self.toggle:
            if self.add or self.remove:
                raise RuleError("Can't mix toggle with add or remove")
            if len(tog) > 1:
                raise RuleError("May only toggle 1 role at once")
        else:
            if not (self.add or self.remove):
                raise RuleError("Must provide at least one of add, remove, or toggle as an action")

# can expand this to a type union if needed later on, until then it's an alias where future API should handle this.
VERSIONED_DATA = DataV1


def pack_rules(data: VERSIONED_DATA, /) -> bytes:
    data.validate()
    version, *ordered_lists = data
    struct_fmt = "!bb%dQb%dQb%dQb%dQb%dQb%dQ" % tuple(map(len, ordered_lists))
    to_pack = chain.from_iterable((len(lst), *lst) for lst in ordered_lists)
    return struct.pack(struct_fmt, version, *to_pack)


def _v1_struct_unpacker(raw: bytes, /) -> Iterator[frozenset[int]]:
    """
    Calling contract is that you have checked the version in advance
    """
    offset = 1
    for _ in range(6):
        (q,) = struct.unpack_from("!b", raw, offset)
        yield frozenset(struct.unpack_from(f"{q}Q", raw, offset + 1))
        offset += 8 * q + 1


def _get_data_version(b: bytes, /) -> int:
    (r,) = struct.unpack("!b", b)
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
        if version != 1:
            raise NoUserFeedback
        data = DataV1(*_v1_struct_unpacker(raw))
        data.validate()
    except (NoUserFeedback, struct.error):
        raise NoUserFeedback from None
    except Exception as exc:
        log.exception("Unhandled exception type %s", type(exc), exc_info=False)
        raise NoUserFeedback from None
    else:
        return data


def get_secret_data_from_file(path: Path) -> tuple[int, AESSIV]:
    """
    We get back a number that represents an equivalent 
    time portion of a discord snowflake (snowflake >> 22)
    for when a key was generated ad well as a key
    """
    with path.open(mode="rb") as fp:
        raw = fp.read()
        ts, k =  struct.unpack("!Q64s", raw)
        return ts, AESSIV(k)


def _generate_secret_data_file(path: Path):
    """
    Writes a file in the format expected by `get_secret_data`
    after generating the values and writes to a path.

    assumes system clock is fine.

    will not overwrite an existing path
    (barring filesytem race conditions not considered in scope to account for)
    """

    if path.exists():
        return

    key = AESSIV.generate_key(512)
    time = discord.utils.time_snowflake(discord.utils.utcnow()) >> 22

    with path.open(mode="rb") as fp:
        fp.write(struct.pack("!Q64s", time, key))


class RoleBot(discord.AutoShardedClient):
    def __init__(self, valid_since: int, aessiv: AESSIV) -> None:
        self.aessiv = aessiv
        self.valid_since = valid_since
        self.interaction_regex = re.compile(r"^rr\d{2}:(.*)$", flags=re.DOTALL)
        self.tree = discord.app_commands.CommandTree(self, fallback_to_global=False)
        super().__init__(intents=discord.Intents(1))

    async def on_interaction(self: Self, interaction: discord.Interaction[Self]):
        if interaction.type is discord.InteractionType.component:
            # components *should* be guaranteed to have a custom_id by discord
            assert interaction.data is not None
            custom_id = interaction.data.get("custom_id", "")
            # above avoids relying on non-publicly exported types in an assertion
            if m := self.interaction_regex.match(custom_id):
                await self.handle_rules_for_interaction(interaction, m.group(1))

    async def handle_rules_for_interaction(
        self: Self, interaction: discord.Interaction[Self], encoded_rules: str
    ):
        await interaction.response.defer(ephemeral=True)

        # calling contract here ensures: 
        # 1. That the message exists 
        # 2. That the guild exists 
        # 3. that the user is a member

        assert interaction.message is not None
        assert isinstance(interaction.user, discord.Member)
        assert interaction.guild is not None

        message = interaction.message
        user = interaction.user
        guild = interaction.guild

        if (message.id >> 22) < self.valid_since:
            log.info("Not processing button that predates key on message_id: %d from user: %d", message.id, user.id)
            return

        try:
            by = base2048.decode(encoded_rules)
        except Exception:
            log.exception(
                "SEC: Could not decode expected valid custom id using base2048: %s from user %d",
                encoded_rules,
                user.id,
            )
            return

        try:
            packed_rules = self.aessiv.decrypt(by, [message.id.to_bytes(8, byteorder="big")])
        except InvalidTag:
            log.exception(
                "SEC: Got a custom id %s that couldn't decrypt with aessiv key from user %d",
                encoded_rules,
                user.id,
            )
            return

        try:
            rules = unpack_rules(packed_rules)
        except NoUserFeedback as exc:
            log.exception(
                "SEC: Could not unpack custom_id %s after decryption passed from user %d",
                encoded_rules,
                user.id,
            )
            return

        starting_role_ids = {r.id for r in user.roles}

        if any(
            (
                (rules.require_any and not (rules.require_any & starting_role_ids)),
                ((rules.require_all & starting_role_ids) - rules.require_all),
                (rules.require_none & starting_role_ids),
            )
        ):
            return await interaction.followup.send(
                content="You are ineligible to use this button", ephemeral=True
            )

        new_role_ids = starting_role_ids

        new_role_ids ^= rules.toggle
        new_role_ids |= rules.add
        new_role_ids -= rules.remove

        # not using symetric difference here because we need these seperately later anyhow
        added, removed = (
            new_role_ids - starting_role_ids,
            starting_role_ids - new_role_ids,
        )

        if not (added or removed):
            return  # nothing to change

        if not guild.me.guild_permissions.manage_roles:
            return await interaction.followup.send(
                content="I no longer have permission to do that", ephemeral=True
            )

        for rid in chain(added, removed):
            if role := guild.get_role(rid):
                if role >= guild.me.top_role:
                    return await interaction.followup.send(
                        content=f"I can't assign a role this button references: {role.mention}",
                        ephemeral=True,
                    )
            else:
                # SEC: There's a balance between security and usability to consider below
                # The ideal for security is to silently fail here too.
                # The reality is that successfully attacking to this point
                # and not tripping prior alarms for an extended period measuring years...
                if (rid.bit_length() < 43) or not (
                    (guild.id >> 22) < (rid >> 22) < (interaction.message.id >> 22)
                ):
                    log.critical(
                        "SEC: passed decryption, passed unpack, impossible ID %d from packed rules %s from user %d",
                        rid,
                        packed_rules,
                        user.id,
                    )
                    return
                else:
                    return await interaction.followup.send(
                        content="This button references a role that no longer exists.",
                        ephemeral=True,
                    )

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
        content: str,
        label_action_pairs: list[tuple[str, VERSIONED_DATA]],
    ):

        if len(label_action_pairs) > 25:
            raise ValueError("Cannot provide that many buttons in a view")

        view = discord.ui.View(timeout=None)
        view.stop()

        message: discord.Message = await channel.send(content=content)

        for idx, (label, data) in enumerate(label_action_pairs):
            packed = pack_rules(data)
            enc = self.aessiv.encrypt(packed, [message.id.to_bytes(8, byteorder="big")])
            to_disc = base2048.encode(enc)
            custom_id = f"rr{idx:02}:{to_disc}"
            button: discord.ui.Button[discord.ui.View] = discord.ui.Button(label=label, custom_id=custom_id)
            view.add_item(button)

        await message.edit(view=view)


# TODO: figure out a "good" way to handle user input creating these buttons statelessly and without additional intents
