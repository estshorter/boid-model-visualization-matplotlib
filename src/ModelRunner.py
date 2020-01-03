import contextlib
import datetime
import logging
from logging.handlers import MemoryHandler
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterator

from matplotlib.animation import Animation, FuncAnimation
import matplotlib.pyplot as plt
from mesa import Model
import toml
from tqdm import tqdm

from TqdmLoggingHandler import TqdmLoggingHandler

logger = logging.getLogger(__name__)


def make_parent_dir(filename: str) -> None:
    parent = Path(filename).parent
    if not parent.exists():
        parent.mkdir()


def initialize_root_logger(desc: str) -> None:
    # add TqdmLoggingHandler and MemoryHandler targeting FileHandler
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    tqdm_handler = TqdmLoggingHandler(level=logging.INFO)
    tqdm_handler.setFormatter(logging.Formatter("{message}", style="{"))
    logger.addHandler(tqdm_handler)

    dt_now = datetime.datetime.now()
    filename = f"./log/{dt_now.strftime('%Y%m%d_%H%M%S')}_{desc}.log"
    make_parent_dir(filename)
    # encodingを指定しないと、Windowsではshift-jisで出力される
    # delay=Trueを指定し、初書込み時にファイルを作成するようにする
    file_handler = logging.FileHandler(filename, encoding="utf-8", delay=True)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("{levelname:<5}| {message}", style="{"))
    # いちいちファイルに書き込むと遅いのでMemoryHandlerを使う
    # logger.info(), logger.debug()などを呼んだ回数がcapacityを上回った場合、targetに出力される
    # flushLevelは高めに設定
    memory_handler = MemoryHandler(
        capacity=100, flushLevel=logging.ERROR, target=file_handler
    )
    logger.addHandler(memory_handler)

    logger.info(f"{desc} @ {dt_now.strftime('%Y/%m/%d %H:%M:%S')}")
    logger.debug(f"Args: {sys.argv}")


def log_elapsed_time(elapsed_sec: float) -> None:
    m, s = divmod(elapsed_sec, 60)
    h, m = divmod(m, 60)
    logger.info(f"Elapsed time: {int(h):02d}:{int(m):02d}:{s:.1f}")


def log_parameters(params) -> None:
    with decorate_print(logger.debug, "Parameter"):
        for line in toml.dumps(params).splitlines():
            logger.debug(line)


@contextlib.contextmanager
def decorate_print(
    print_func: Callable[[str], None],
    string: str,
    char_deco: str = "=",
    len_deco: int = 35,
) -> Iterator[None]:
    print_func(f" {string} ".center(len_deco, char_deco))
    yield
    print_func(char_deco * len_deco)


class FuncAnimationWithEndFunc(FuncAnimation):
    def __init__(
        self,
        fig,
        func,
        frames=None,
        init_func=None,
        fargs=None,
        save_count=None,
        *,
        end_func,
        cache_frame_data=True,
        **kwargs,
    ):
        super().__init__(
            fig,
            func,
            frames,
            init_func,
            fargs,
            save_count,
            cache_frame_data=cache_frame_data,
            **kwargs,
        )
        self._end_func = end_func

    def _step(self, *args):
        still_going = Animation._step(self, *args)
        if not still_going:
            # If self._end_func includes plt.close, returning False raises an exception
            # So, belows are workaround
            self.event_source.remove_callback(self._step)
            self._end_func()
        return True


class ModelRunner:
    def __init__(self, model: Model, param_path: str) -> None:
        params = toml.load(param_path)
        self.model = model(**params["model"])
        self.params = params
        pglo = params["global"]
        initialize_root_logger(pglo["description"])
        self.max_timestep = pglo["max_timestep"]

    def update(self, iter: int, pbar: tqdm) -> None:
        self.model.step()
        self.model.draw_succesive()
        pbar.update(1)

    def run(self, callback: Callable[[FuncAnimation], Any]) -> None:
        start = time.monotonic()
        self.model.draw_initial()
        with tqdm(total=self.max_timestep) as pbar:
            fanm = FuncAnimationWithEndFunc(
                self.model.fig,
                self.update,
                fargs=(pbar,),
                interval=self.interval,
                frames=self.max_timestep - 1,
                end_func=plt.close,
            )
            callback(fanm)
        with decorate_print(logger.info, "Final Results"):
            log_elapsed_time(time.monotonic() - start)
        log_parameters(self.params)

    def run_headless(self) -> None:
        def consume_iter_gen(fanm: FuncAnimation) -> None:
            for step in fanm._iter_gen():
                fanm._func(step, *fanm._args)

        self.interval = 0
        self.run(consume_iter_gen)

    def run_silent(self) -> None:
        for step in range(self.max_timestep):
            self.model.step()

    def visualize(self) -> None:
        self.interval = self.params["visualization"]["interval"]
        self.run(lambda fanm: plt.show())

    def save(self, filename: str, writer: str = "ffmpeg") -> None:
        make_parent_dir(filename)
        pmovie = self.params["movie"]
        self.interval = pmovie["interval"]
        self.run(lambda fanm: fanm.save(filename, writer, dpi=pmovie["dpi"]))


if __name__ == "__main__":
    from model import BoidFlockers

    param_path = "./parameter/nominal.toml"
    runner = ModelRunner(BoidFlockers, param_path)
    runner.visualize()
