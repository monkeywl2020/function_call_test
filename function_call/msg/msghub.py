# -*- coding: utf-8 -*-
"""MsgHub is designed to share messages among a group of bots.
"""
from __future__ import annotations
from typing import Any, Optional, Union, Sequence
import logging

#from ..agents.operator import Operator
from ..logger.logfun import LOG_INFO,LOG_DEBUG,LOG_WARNING,LOG_WARNING,LOG_ERROR,LOG_CRITICAL


class MsgHubManager:
    """MsgHub manager class for sharing dialog among a group of bots."""

    def __init__(
        self,
        participants: Sequence[Operator],
        announcement: Optional[Union[Sequence[dict], dict]] = None,
    ) -> None:
        """Initialize a msghub manager from the given arguments.

        Args:
            participants (`Sequence[BotBase]`):
                The Sequence of participants in the msghub.
            announcement
                (`Optional[Union[list[dict], dict]]`, defaults to `None`):
                The message that will be broadcast to all participants at
                the first without requiring response.
        """
        self.logger = logging.getLogger(__name__)  # 获取日志记录器实例

        #participants 是一个 BotBase 序列
        self.participants = participants
        # 进入 msghub的首次公告，无需应答
        self.announcement = announcement

    def __enter__(self) -> MsgHubManager:
        """Will be called when entering the msghub."""
        name_participants = [bot.name for bot in self.participants]
        LOG_DEBUG(self.logger,
            "Enter msghub with participants: {}",
            ", ".join(
                name_participants,
            ),
        )

        self._reset_audience()

        # broadcast the input message to all participants 接收公告
        if self.announcement is not None:
            for bot in self.participants:
                bot.observe(self.announcement)

        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        """Will be called when exiting the msghub."""
        #for bot in self.participants:
        #   bot.clear_audience()

    def _reset_audience(self) -> None:
        """Reset the audience for bot in `self.participant`"""
        # 暂时不出来 bot内部的内容
        #for bot in self.participants:
        #    bot.reset_audience(self.participants)

    # 添加新成员
    def add(
        self,
        new_participant: Union[Sequence[Operator], Operator],
    ) -> None:
        """Add new participant into this hub"""
        if isinstance(new_participant, Operator):
            new_participant = [new_participant]

        # 新进的bot成员
        for bot in new_participant:
            if bot not in self.participants:
                self.participants.append(bot)
            else:
                LOG_WARNING(self.logger,
                    f"Skip adding bot [{bot.name}] for it has "
                    "already joined in.",
                )

        self._reset_audience()

    # 删除新成员
    def delete(
        self,
        participant: Union[Sequence[Operator], Operator],
    ) -> None:
        """Delete bots from participant."""
        if isinstance(participant, Operator):
            participant = [participant]

        for bot in participant:
            if bot in self.participants:
                # Clear the audience of the deleted bot firstly
                #bot.clear_audience()

                # remove bot from self.participant
                self.participants.pop(self.participants.index(bot))
            else:
                LOG_WARNING(self.logger,
                    f"Cannot find bot [{bot.name}], skip its"
                    f" deletion.",
                )

        # Remove this bot from the audience of other bots
        self._reset_audience()

    def broadcast(self, msg: Union[dict, list[dict]]) -> None:
        """Broadcast the message to all participants.

        Args:
            msg (`Union[dict, list[dict]]`):
                One or a list of dict messages to broadcast among all
                participants.
        """
        for bot in self.participants:
            bot.observe(msg)


def msghub(
    participants: Sequence[Operator],
    announcement: Optional[Union[Sequence[dict], dict]] = None,
) -> MsgHubManager:
    """msghub is used to share messages among a group of bots.

    Args:
        participants (`Sequence[AgentBase]`):
            A Sequence of participated bots in the msghub.
        announcement (`Optional[Union[list[dict], dict]]`, defaults to `None`):
            The message that will be broadcast to all participants at the
            very beginning without requiring response.

    Example:
        In the following code, we create a msghub with three bots, and each
        message output by `bot1`, `bot2`, `bot3` will be passed to all
        other bots, that's what we mean msghub.

        .. code-block:: python

            with msghub(participant=[bot1, bot2, bot3]):
                bot1()
                bot2()

        Actually, it has the same effect as the following code, but much more
        easy and elegant!

        .. code-block:: python

            x1 = bot1()
            bot2.observe(x1)
            bot3.observe(x1)

            x2 = bot2()
            bot1.observe(x2)
            bot3.observe(x2)
    """
    return MsgHubManager(participants, announcement)
