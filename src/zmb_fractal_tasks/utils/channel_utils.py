""""Utilities for channel selection."""

from typing import Literal

from ngio import ChannelSelectionModel
from pydantic import BaseModel, Field


class MeasurementChannels(BaseModel):
    """Channel configuration.

    Args:
        This model is used to select a channel by label, wavelength ID, or index.

    Args:
        use_all_channels (bool): If True, all channels are selected.
        mode (Literal["label", "wavelength_id", "index"]): Specifies how to
            interpret the identifier. Can be "label", "wavelength_id", or
            "index" (must be an integer).
        identifiers (list[str]): Unique identifier for the channel.
            This can be a channel label, wavelength ID, or index.

    """

    use_all_channels: bool = True
    mode: Literal["label", "wavelength_id", "index"] = "label"
    identifiers: list[str] = Field(default_factory=list)

    def to_list(self) -> list[ChannelSelectionModel]:
        """Convert to list of ChannelSelectionModel.

        Returns:
            list[ChannelSelectionModel]: List of ChannelSelectionModel.
        """
        return [
            ChannelSelectionModel(identifier=identifier, mode=self.mode)
            for identifier in self.identifiers
        ]