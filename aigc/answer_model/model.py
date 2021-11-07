"""AIGC2021 submission model.

Note: This is auto-generated .py DO NOT modify.
"""

import torch
from torch import nn

framework = "torch"  # type: ignore


class CompressionModel(nn.Module):  # type: ignore
    """CompressedModel for AIGC2021."""

    def __init__(self) -> None:  # type: ignore
        """Initialize model."""
        super().__init__()
        self.module_003 = nn.Conv2d(3, 64, kernel_size=(6, 6), stride=(2, 2), padding=(2, 2))  # type: ignore
        self.module_004 = nn.SiLU()  # type: ignore
        self.module_007 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))  # type: ignore
        self.module_008 = nn.SiLU()  # type: ignore
        self.module_012 = nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_013 = nn.SiLU()  # type: ignore
        self.module_016 = nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_017 = nn.SiLU()  # type: ignore
        self.module_020 = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_021 = nn.SiLU()  # type: ignore
        self.module_026 = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_027 = nn.SiLU()  # type: ignore
        self.module_030 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  # type: ignore
        self.module_031 = nn.SiLU()  # type: ignore
        self.module_036 = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_037 = nn.SiLU()  # type: ignore
        self.module_040 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  # type: ignore
        self.module_041 = nn.SiLU()  # type: ignore
        self.module_046 = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_047 = nn.SiLU()  # type: ignore
        self.module_050 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  # type: ignore
        self.module_051 = nn.SiLU()  # type: ignore
        self.module_057 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))  # type: ignore
        self.module_058 = nn.SiLU()  # type: ignore
        self.module_062 = nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_063 = nn.SiLU()  # type: ignore
        self.module_066 = nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_067 = nn.SiLU()  # type: ignore
        self.module_070 = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_071 = nn.SiLU()  # type: ignore
        self.module_076 = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_077 = nn.SiLU()  # type: ignore
        self.module_080 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  # type: ignore
        self.module_081 = nn.SiLU()  # type: ignore
        self.module_086 = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_087 = nn.SiLU()  # type: ignore
        self.module_090 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  # type: ignore
        self.module_091 = nn.SiLU()  # type: ignore
        self.module_096 = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_097 = nn.SiLU()  # type: ignore
        self.module_100 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  # type: ignore
        self.module_101 = nn.SiLU()  # type: ignore
        self.module_106 = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_107 = nn.SiLU()  # type: ignore
        self.module_110 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  # type: ignore
        self.module_111 = nn.SiLU()  # type: ignore
        self.module_116 = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_117 = nn.SiLU()  # type: ignore
        self.module_120 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  # type: ignore
        self.module_121 = nn.SiLU()  # type: ignore
        self.module_126 = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_127 = nn.SiLU()  # type: ignore
        self.module_130 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  # type: ignore
        self.module_131 = nn.SiLU()  # type: ignore
        self.module_137 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))  # type: ignore
        self.module_138 = nn.SiLU()  # type: ignore
        self.module_142 = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_143 = nn.SiLU()  # type: ignore
        self.module_146 = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_147 = nn.SiLU()  # type: ignore
        self.module_150 = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_151 = nn.SiLU()  # type: ignore
        self.module_156 = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_157 = nn.SiLU()  # type: ignore
        self.module_160 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  # type: ignore
        self.module_161 = nn.SiLU()  # type: ignore
        self.module_166 = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_167 = nn.SiLU()  # type: ignore
        self.module_170 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  # type: ignore
        self.module_171 = nn.SiLU()  # type: ignore
        self.module_176 = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_177 = nn.SiLU()  # type: ignore
        self.module_180 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  # type: ignore
        self.module_181 = nn.SiLU()  # type: ignore
        self.module_186 = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_187 = nn.SiLU()  # type: ignore
        self.module_190 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  # type: ignore
        self.module_191 = nn.SiLU()  # type: ignore
        self.module_196 = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_197 = nn.SiLU()  # type: ignore
        self.module_200 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  # type: ignore
        self.module_201 = nn.SiLU()  # type: ignore
        self.module_206 = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_207 = nn.SiLU()  # type: ignore
        self.module_210 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  # type: ignore
        self.module_211 = nn.SiLU()  # type: ignore
        self.module_216 = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_217 = nn.SiLU()  # type: ignore
        self.module_220 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  # type: ignore
        self.module_221 = nn.SiLU()  # type: ignore
        self.module_226 = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_227 = nn.SiLU()  # type: ignore
        self.module_230 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  # type: ignore
        self.module_231 = nn.SiLU()  # type: ignore
        self.module_236 = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_237 = nn.SiLU()  # type: ignore
        self.module_240 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  # type: ignore
        self.module_241 = nn.SiLU()  # type: ignore
        self.module_248 = nn.Conv2d(512, 216, kernel_size=(1, 1), stride=(1, 1), bias=False)  # type: ignore
        self.module_249 = nn.Conv2d(216, 288, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)  # type: ignore
        self.module_250 = nn.Conv2d(288, 1024, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_252 = nn.SiLU()  # type: ignore
        self.module_256 = nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_257 = nn.SiLU()  # type: ignore
        self.module_260 = nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_261 = nn.SiLU()  # type: ignore
        self.module_264 = nn.Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_265 = nn.SiLU()  # type: ignore
        self.module_270 = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_271 = nn.SiLU()  # type: ignore
        self.module_275 = nn.Conv2d(512, 139, kernel_size=(1, 1), stride=(1, 1), bias=False)  # type: ignore
        self.module_276 = nn.Conv2d(139, 187, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  # type: ignore
        self.module_277 = nn.Conv2d(187, 512, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_279 = nn.SiLU()  # type: ignore
        self.module_284 = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_285 = nn.SiLU()  # type: ignore
        self.module_289 = nn.Conv2d(512, 167, kernel_size=(1, 1), stride=(1, 1), bias=False)  # type: ignore
        self.module_290 = nn.Conv2d(167, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  # type: ignore
        self.module_291 = nn.Conv2d(192, 512, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_293 = nn.SiLU()  # type: ignore
        self.module_298 = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_299 = nn.SiLU()  # type: ignore
        self.module_303 = nn.Conv2d(512, 171, kernel_size=(1, 1), stride=(1, 1), bias=False)  # type: ignore
        self.module_304 = nn.Conv2d(171, 184, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  # type: ignore
        self.module_305 = nn.Conv2d(184, 512, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_307 = nn.SiLU()  # type: ignore
        self.module_314 = nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_315 = nn.SiLU()  # type: ignore
        self.module_318 = nn.Conv2d(2048, 1024, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_319 = nn.ReLU(inplace=True)  # type: ignore
        self.module_321 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)  # type: ignore
        self.module_324 = nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_325 = nn.SiLU()  # type: ignore
        self.module_327 = nn.Upsample(scale_factor=2.0, mode="nearest")  # type: ignore
        self.module_331 = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_332 = nn.SiLU()  # type: ignore
        self.module_335 = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_336 = nn.SiLU()  # type: ignore
        self.module_339 = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_340 = nn.SiLU()  # type: ignore
        self.module_345 = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_346 = nn.SiLU()  # type: ignore
        self.module_349 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  # type: ignore
        self.module_350 = nn.SiLU()  # type: ignore
        self.module_355 = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_356 = nn.SiLU()  # type: ignore
        self.module_359 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  # type: ignore
        self.module_360 = nn.SiLU()  # type: ignore
        self.module_365 = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_366 = nn.SiLU()  # type: ignore
        self.module_369 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  # type: ignore
        self.module_370 = nn.SiLU()  # type: ignore
        self.module_376 = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_377 = nn.SiLU()  # type: ignore
        self.module_379 = nn.Upsample(scale_factor=2.0, mode="nearest")  # type: ignore
        self.module_383 = nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_384 = nn.SiLU()  # type: ignore
        self.module_387 = nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_388 = nn.SiLU()  # type: ignore
        self.module_391 = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_392 = nn.SiLU()  # type: ignore
        self.module_397 = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_398 = nn.SiLU()  # type: ignore
        self.module_401 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  # type: ignore
        self.module_402 = nn.SiLU()  # type: ignore
        self.module_407 = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_408 = nn.SiLU()  # type: ignore
        self.module_411 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  # type: ignore
        self.module_412 = nn.SiLU()  # type: ignore
        self.module_417 = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_418 = nn.SiLU()  # type: ignore
        self.module_421 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  # type: ignore
        self.module_422 = nn.SiLU()  # type: ignore
        self.module_429 = nn.Conv2d(256, 114, kernel_size=(1, 1), stride=(1, 1), bias=False)  # type: ignore
        self.module_430 = nn.Conv2d(114, 114, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)  # type: ignore
        self.module_431 = nn.Conv2d(114, 256, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_433 = nn.SiLU()  # type: ignore
        self.module_438 = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_439 = nn.SiLU()  # type: ignore
        self.module_442 = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_443 = nn.SiLU()  # type: ignore
        self.module_446 = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_447 = nn.SiLU()  # type: ignore
        self.module_452 = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_453 = nn.SiLU()  # type: ignore
        self.module_457 = nn.Conv2d(256, 92, kernel_size=(1, 1), stride=(1, 1), bias=False)  # type: ignore
        self.module_458 = nn.Conv2d(92, 104, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  # type: ignore
        self.module_459 = nn.Conv2d(104, 256, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_461 = nn.SiLU()  # type: ignore
        self.module_466 = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_467 = nn.SiLU()  # type: ignore
        self.module_470 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  # type: ignore
        self.module_471 = nn.SiLU()  # type: ignore
        self.module_476 = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_477 = nn.SiLU()  # type: ignore
        self.module_481 = nn.Conv2d(256, 82, kernel_size=(1, 1), stride=(1, 1), bias=False)  # type: ignore
        self.module_482 = nn.Conv2d(82, 78, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  # type: ignore
        self.module_483 = nn.Conv2d(78, 256, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_485 = nn.SiLU()  # type: ignore
        self.module_492 = nn.Conv2d(512, 117, kernel_size=(1, 1), stride=(1, 1), bias=False)  # type: ignore
        self.module_493 = nn.Conv2d(117, 157, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)  # type: ignore
        self.module_494 = nn.Conv2d(157, 512, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_496 = nn.SiLU()  # type: ignore
        self.module_501 = nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_502 = nn.SiLU()  # type: ignore
        self.module_505 = nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_506 = nn.SiLU()  # type: ignore
        self.module_509 = nn.Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_510 = nn.SiLU()  # type: ignore
        self.module_515 = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_516 = nn.SiLU()  # type: ignore
        self.module_520 = nn.Conv2d(512, 114, kernel_size=(1, 1), stride=(1, 1), bias=False)  # type: ignore
        self.module_521 = nn.Conv2d(114, 134, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  # type: ignore
        self.module_522 = nn.Conv2d(134, 512, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_524 = nn.SiLU()  # type: ignore
        self.module_529 = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_530 = nn.SiLU()  # type: ignore
        self.module_534 = nn.Conv2d(512, 143, kernel_size=(1, 1), stride=(1, 1), bias=False)  # type: ignore
        self.module_535 = nn.Conv2d(143, 139, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  # type: ignore
        self.module_536 = nn.Conv2d(139, 512, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_538 = nn.SiLU()  # type: ignore
        self.module_543 = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_544 = nn.SiLU()  # type: ignore
        self.module_548 = nn.Conv2d(512, 137, kernel_size=(1, 1), stride=(1, 1), bias=False)  # type: ignore
        self.module_549 = nn.Conv2d(137, 139, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  # type: ignore
        self.module_550 = nn.Conv2d(139, 512, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_552 = nn.SiLU()  # type: ignore
        self.module_559 = nn.Conv2d(256, 255, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_560 = nn.Conv2d(512, 255, kernel_size=(1, 1), stride=(1, 1))  # type: ignore
        self.module_561 = nn.Conv2d(1024, 255, kernel_size=(1, 1), stride=(1, 1))  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Run the model.

        Caution: This method will not work since its purpose
        is to compute number of parameters of the model.
        """
        x = self.module_003(x)  # type: ignore
        x = self.module_004(x)  # type: ignore
        x = self.module_007(x)  # type: ignore
        x = self.module_008(x)  # type: ignore
        x = self.module_012(x)  # type: ignore
        x = self.module_013(x)  # type: ignore
        x = self.module_016(x)  # type: ignore
        x = self.module_017(x)  # type: ignore
        x = self.module_020(x)  # type: ignore
        x = self.module_021(x)  # type: ignore
        x = self.module_026(x)  # type: ignore
        x = self.module_027(x)  # type: ignore
        x = self.module_030(x)  # type: ignore
        x = self.module_031(x)  # type: ignore
        x = self.module_036(x)  # type: ignore
        x = self.module_037(x)  # type: ignore
        x = self.module_040(x)  # type: ignore
        x = self.module_041(x)  # type: ignore
        x = self.module_046(x)  # type: ignore
        x = self.module_047(x)  # type: ignore
        x = self.module_050(x)  # type: ignore
        x = self.module_051(x)  # type: ignore
        x = self.module_057(x)  # type: ignore
        x = self.module_058(x)  # type: ignore
        x = self.module_062(x)  # type: ignore
        x = self.module_063(x)  # type: ignore
        x = self.module_066(x)  # type: ignore
        x = self.module_067(x)  # type: ignore
        x = self.module_070(x)  # type: ignore
        x = self.module_071(x)  # type: ignore
        x = self.module_076(x)  # type: ignore
        x = self.module_077(x)  # type: ignore
        x = self.module_080(x)  # type: ignore
        x = self.module_081(x)  # type: ignore
        x = self.module_086(x)  # type: ignore
        x = self.module_087(x)  # type: ignore
        x = self.module_090(x)  # type: ignore
        x = self.module_091(x)  # type: ignore
        x = self.module_096(x)  # type: ignore
        x = self.module_097(x)  # type: ignore
        x = self.module_100(x)  # type: ignore
        x = self.module_101(x)  # type: ignore
        x = self.module_106(x)  # type: ignore
        x = self.module_107(x)  # type: ignore
        x = self.module_110(x)  # type: ignore
        x = self.module_111(x)  # type: ignore
        x = self.module_116(x)  # type: ignore
        x = self.module_117(x)  # type: ignore
        x = self.module_120(x)  # type: ignore
        x = self.module_121(x)  # type: ignore
        x = self.module_126(x)  # type: ignore
        x = self.module_127(x)  # type: ignore
        x = self.module_130(x)  # type: ignore
        x = self.module_131(x)  # type: ignore
        x = self.module_137(x)  # type: ignore
        x = self.module_138(x)  # type: ignore
        x = self.module_142(x)  # type: ignore
        x = self.module_143(x)  # type: ignore
        x = self.module_146(x)  # type: ignore
        x = self.module_147(x)  # type: ignore
        x = self.module_150(x)  # type: ignore
        x = self.module_151(x)  # type: ignore
        x = self.module_156(x)  # type: ignore
        x = self.module_157(x)  # type: ignore
        x = self.module_160(x)  # type: ignore
        x = self.module_161(x)  # type: ignore
        x = self.module_166(x)  # type: ignore
        x = self.module_167(x)  # type: ignore
        x = self.module_170(x)  # type: ignore
        x = self.module_171(x)  # type: ignore
        x = self.module_176(x)  # type: ignore
        x = self.module_177(x)  # type: ignore
        x = self.module_180(x)  # type: ignore
        x = self.module_181(x)  # type: ignore
        x = self.module_186(x)  # type: ignore
        x = self.module_187(x)  # type: ignore
        x = self.module_190(x)  # type: ignore
        x = self.module_191(x)  # type: ignore
        x = self.module_196(x)  # type: ignore
        x = self.module_197(x)  # type: ignore
        x = self.module_200(x)  # type: ignore
        x = self.module_201(x)  # type: ignore
        x = self.module_206(x)  # type: ignore
        x = self.module_207(x)  # type: ignore
        x = self.module_210(x)  # type: ignore
        x = self.module_211(x)  # type: ignore
        x = self.module_216(x)  # type: ignore
        x = self.module_217(x)  # type: ignore
        x = self.module_220(x)  # type: ignore
        x = self.module_221(x)  # type: ignore
        x = self.module_226(x)  # type: ignore
        x = self.module_227(x)  # type: ignore
        x = self.module_230(x)  # type: ignore
        x = self.module_231(x)  # type: ignore
        x = self.module_236(x)  # type: ignore
        x = self.module_237(x)  # type: ignore
        x = self.module_240(x)  # type: ignore
        x = self.module_241(x)  # type: ignore
        x = self.module_248(x)  # type: ignore
        x = self.module_249(x)  # type: ignore
        x = self.module_250(x)  # type: ignore
        x = self.module_252(x)  # type: ignore
        x = self.module_256(x)  # type: ignore
        x = self.module_257(x)  # type: ignore
        x = self.module_260(x)  # type: ignore
        x = self.module_261(x)  # type: ignore
        x = self.module_264(x)  # type: ignore
        x = self.module_265(x)  # type: ignore
        x = self.module_270(x)  # type: ignore
        x = self.module_271(x)  # type: ignore
        x = self.module_275(x)  # type: ignore
        x = self.module_276(x)  # type: ignore
        x = self.module_277(x)  # type: ignore
        x = self.module_279(x)  # type: ignore
        x = self.module_284(x)  # type: ignore
        x = self.module_285(x)  # type: ignore
        x = self.module_289(x)  # type: ignore
        x = self.module_290(x)  # type: ignore
        x = self.module_291(x)  # type: ignore
        x = self.module_293(x)  # type: ignore
        x = self.module_298(x)  # type: ignore
        x = self.module_299(x)  # type: ignore
        x = self.module_303(x)  # type: ignore
        x = self.module_304(x)  # type: ignore
        x = self.module_305(x)  # type: ignore
        x = self.module_307(x)  # type: ignore
        x = self.module_314(x)  # type: ignore
        x = self.module_315(x)  # type: ignore
        x = self.module_318(x)  # type: ignore
        x = self.module_319(x)  # type: ignore
        x = self.module_321(x)  # type: ignore
        x = self.module_324(x)  # type: ignore
        x = self.module_325(x)  # type: ignore
        x = self.module_327(x)  # type: ignore
        x = self.module_331(x)  # type: ignore
        x = self.module_332(x)  # type: ignore
        x = self.module_335(x)  # type: ignore
        x = self.module_336(x)  # type: ignore
        x = self.module_339(x)  # type: ignore
        x = self.module_340(x)  # type: ignore
        x = self.module_345(x)  # type: ignore
        x = self.module_346(x)  # type: ignore
        x = self.module_349(x)  # type: ignore
        x = self.module_350(x)  # type: ignore
        x = self.module_355(x)  # type: ignore
        x = self.module_356(x)  # type: ignore
        x = self.module_359(x)  # type: ignore
        x = self.module_360(x)  # type: ignore
        x = self.module_365(x)  # type: ignore
        x = self.module_366(x)  # type: ignore
        x = self.module_369(x)  # type: ignore
        x = self.module_370(x)  # type: ignore
        x = self.module_376(x)  # type: ignore
        x = self.module_377(x)  # type: ignore
        x = self.module_379(x)  # type: ignore
        x = self.module_383(x)  # type: ignore
        x = self.module_384(x)  # type: ignore
        x = self.module_387(x)  # type: ignore
        x = self.module_388(x)  # type: ignore
        x = self.module_391(x)  # type: ignore
        x = self.module_392(x)  # type: ignore
        x = self.module_397(x)  # type: ignore
        x = self.module_398(x)  # type: ignore
        x = self.module_401(x)  # type: ignore
        x = self.module_402(x)  # type: ignore
        x = self.module_407(x)  # type: ignore
        x = self.module_408(x)  # type: ignore
        x = self.module_411(x)  # type: ignore
        x = self.module_412(x)  # type: ignore
        x = self.module_417(x)  # type: ignore
        x = self.module_418(x)  # type: ignore
        x = self.module_421(x)  # type: ignore
        x = self.module_422(x)  # type: ignore
        x = self.module_429(x)  # type: ignore
        x = self.module_430(x)  # type: ignore
        x = self.module_431(x)  # type: ignore
        x = self.module_433(x)  # type: ignore
        x = self.module_438(x)  # type: ignore
        x = self.module_439(x)  # type: ignore
        x = self.module_442(x)  # type: ignore
        x = self.module_443(x)  # type: ignore
        x = self.module_446(x)  # type: ignore
        x = self.module_447(x)  # type: ignore
        x = self.module_452(x)  # type: ignore
        x = self.module_453(x)  # type: ignore
        x = self.module_457(x)  # type: ignore
        x = self.module_458(x)  # type: ignore
        x = self.module_459(x)  # type: ignore
        x = self.module_461(x)  # type: ignore
        x = self.module_466(x)  # type: ignore
        x = self.module_467(x)  # type: ignore
        x = self.module_470(x)  # type: ignore
        x = self.module_471(x)  # type: ignore
        x = self.module_476(x)  # type: ignore
        x = self.module_477(x)  # type: ignore
        x = self.module_481(x)  # type: ignore
        x = self.module_482(x)  # type: ignore
        x = self.module_483(x)  # type: ignore
        x = self.module_485(x)  # type: ignore
        x = self.module_492(x)  # type: ignore
        x = self.module_493(x)  # type: ignore
        x = self.module_494(x)  # type: ignore
        x = self.module_496(x)  # type: ignore
        x = self.module_501(x)  # type: ignore
        x = self.module_502(x)  # type: ignore
        x = self.module_505(x)  # type: ignore
        x = self.module_506(x)  # type: ignore
        x = self.module_509(x)  # type: ignore
        x = self.module_510(x)  # type: ignore
        x = self.module_515(x)  # type: ignore
        x = self.module_516(x)  # type: ignore
        x = self.module_520(x)  # type: ignore
        x = self.module_521(x)  # type: ignore
        x = self.module_522(x)  # type: ignore
        x = self.module_524(x)  # type: ignore
        x = self.module_529(x)  # type: ignore
        x = self.module_530(x)  # type: ignore
        x = self.module_534(x)  # type: ignore
        x = self.module_535(x)  # type: ignore
        x = self.module_536(x)  # type: ignore
        x = self.module_538(x)  # type: ignore
        x = self.module_543(x)  # type: ignore
        x = self.module_544(x)  # type: ignore
        x = self.module_548(x)  # type: ignore
        x = self.module_549(x)  # type: ignore
        x = self.module_550(x)  # type: ignore
        x = self.module_552(x)  # type: ignore
        x = self.module_559(x)  # type: ignore
        x = self.module_560(x)  # type: ignore
        x = self.module_561(x)  # type: ignore
