# SPDX-License-Identifier: Apache-2.0

import os

import gguf
import numpy as np

from vllm_gguf_plugin.gguf_reader import get_gguf_reader


def write_gguf(path, value):
    writer = gguf.GGUFWriter(str(path), "llama")
    writer.add_uint32("test.value", value)
    writer.add_tensor("blk.0.weight", np.zeros((4, 8), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_same_file_is_parsed_once(tmp_path):
    path = tmp_path / "model.gguf"
    write_gguf(path, 1)
    link = tmp_path / "link.gguf"
    link.symlink_to(path)

    reader = get_gguf_reader(path)
    assert get_gguf_reader(str(path)) is reader
    assert get_gguf_reader(link) is reader
    assert [t.name for t in reader.tensors] == ["blk.0.weight"]


def test_changed_file_is_parsed_again(tmp_path):
    path = tmp_path / "model.gguf"
    write_gguf(path, 1)
    first = get_gguf_reader(path)

    write_gguf(path, 22222)
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    second = get_gguf_reader(path)

    assert second is not first
    field = second.fields["test.value"]
    assert int(field.parts[field.data[0]][0]) == 22222
