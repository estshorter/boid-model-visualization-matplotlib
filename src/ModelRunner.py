import contextlib
import datetime
import logging
from logging.handlers import MemoryHandler
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterator

import matplotlib.animation as anm
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


class ModelRunner:
    def __init__(self, model: Model, param_path: str) -> None:
        params = toml.load(param_path)
        self.model = model(**params["model"])
        self.params = params
        ModelRunner.initialize_root_logger(params["global"]["description"])

    def update(self, iter: int, max_timestep: int, pbar: tqdm) -> None:
        self.model.step()
        self.model.draw_succesive()
        pbar.update(1)
        if iter >= max_timestep - 1:
            plt.close()

    def run(self, callback: Callable[[anm.FuncAnimation], Any]) -> None:
        max_timestep = self.params["global"]["max_timestep"]
        start = time.monotonic()
        self.model.draw_initial()
        with tqdm(total=max_timestep) as pbar:
            ani = anm.FuncAnimation(
                self.model.fig,
                self.update,
                fargs=(max_timestep, pbar),
                interval=self.interval,
                frames=max_timestep,
                repeat=False,
            )
            callback(ani)
        with decorate_print(logging.info, "Final Results"):
            ModelRunner.log_elapsed_time(time.monotonic() - start)

    def run_silent(self) -> None:
        def consume_iter_gen(anm: anm.FuncAnimation) -> None:
            for step in anm._iter_gen():
                anm._func(step, *anm._args)

        self.interval = 0
        self.run(consume_iter_gen)

    def visualize(self) -> None:
        self.interval = self.params["visualization"]["interval"]
        self.run(lambda ani: plt.show())

    def save(self, filename: str, writer: str = "ffmpeg") -> None:
        make_parent_dir(filename)
        self.interval = self.params["movie"]["interval"]
        self.run(
            lambda ani: ani.save(
                filename, writer=writer, dpi=self.params["movie"]["dpi"]
            )
        )

    @staticmethod
    def initialize_root_logger(desc: str) -> None:
        # add TqdmLoggingHandler and MemoryHandler targeting FileHandler
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        logging.getLogger("matplotlib").setLevel(logging.CRITICAL)

        formatter_tqdm = logging.Formatter("{message}", style="{")
        tqdm_handler = TqdmLoggingHandler(level=logging.INFO)
        tqdm_handler.setFormatter(formatter_tqdm)
        logger.addHandler(tqdm_handler)

        dt_now = datetime.datetime.now()
        dt_now_str = dt_now.strftime("%Y%m%d_%H%M%S")
        filename = f"./log/{dt_now_str}_{desc}.log"
        make_parent_dir(filename)
        # encodingを指定しないと、Windowsではshift-jisで出力される
        # delay=Trueを指定し、初書込み時にファイルを作成するようにする
        file_handler = logging.FileHandler(filename, encoding="utf-8", delay=True)
        file_handler.setLevel(logging.DEBUG)
        formatter_file = logging.Formatter("{levelname:<5}| {message}", style="{")
        file_handler.setFormatter(formatter_file)
        # いちいちファイルに書き込むと遅いのでMemoryHandlerを使う
        # logging.info(), logging.debug()などを呼んだ回数がcapacityを上回った場合、targetに出力される
        # flushLevelは高めに設定
        memory_handler = MemoryHandler(
            capacity=100, flushLevel=logging.ERROR, target=file_handler
        )
        logger.addHandler(memory_handler)

        logging.info(f"{desc} @ {dt_now.strftime('%Y/%m/%d %H:%M:%S')}")
        logging.debug(f"Args: {sys.argv}")

    @staticmethod
    def log_elapsed_time(elapsed_sec: float) -> None:
        m, s = divmod(elapsed_sec, 60)
        h, m = divmod(m, 60)
        logging.info(f"Elapsed time: {int(h):02d}:{int(m):02d}:{s:.1f}")

    def log_parameters(self) -> None:
        with decorate_print(logging.debug, "Parameter"):
            for line in toml.dumps(self.params).splitlines():
                logging.debug(line)


if __name__ == "__main__":
    from model import BoidFlockers

    param_path = "./parameter/nominal.toml"
    runner = ModelRunner(BoidFlockers, param_path)
    runner.visualize()
    runner.log_parameters()
