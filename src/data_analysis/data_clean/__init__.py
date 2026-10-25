"""Align public tactile datasets onto this project's LeRobot v3 MP4 layout."""

from data_analysis.data_clean.feelsight import FeelSightAligner
from data_analysis.data_clean.freetacman import FreeTacManAligner
from data_analysis.data_clean.haptile import HapTileAligner
from data_analysis.data_clean.tactile_video_pretrain_nc import TactileVideoPretrainAligner
from data_analysis.data_clean.touch_and_go import TouchAndGoAligner
from data_analysis.data_clean.wirefishing import WireFishingAligner

ALIGNERS = {
    HapTileAligner.name: HapTileAligner,
    FreeTacManAligner.name: FreeTacManAligner,
    FeelSightAligner.name: FeelSightAligner,
    WireFishingAligner.name: WireFishingAligner,
    TouchAndGoAligner.name: TouchAndGoAligner,
    TactileVideoPretrainAligner.name: TactileVideoPretrainAligner,
}
