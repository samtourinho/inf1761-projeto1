from __future__ import annotations

from typing import TYPE_CHECKING

import math

import numpy as np
import wgpu
from shape import Shape

if TYPE_CHECKING:
  from state import State

class Disk (Shape):
  """Disco de raio 1 centrado na origem, com coordenadas de textura."""

  def __init__ (self, device: wgpu.GPUDevice, nslices: int = 64) -> None:
    # centro + nslices+1 vertices na borda (o ultimo repete o primeiro)
    coord = [[0.0, 0.0]]
    texcoord = [[0.5, 0.5]]
    for i in range(nslices + 1):
      th = 2 * math.pi * i / nslices
      x, y = math.cos(th), math.sin(th)
      coord.append([x, y])
      texcoord.append([0.5 + 0.5*x, 0.5 - 0.5*y])
    index = []
    for i in range(nslices):
      index += [0, i + 1, i + 2]

    self.nind: int = len(index)
    self.coord_vbo: wgpu.GPUBuffer = device.create_buffer_with_data(data=np.array(coord, dtype='float32'), usage=wgpu.BufferUsage.VERTEX)
    self.texcoord_vbo: wgpu.GPUBuffer = device.create_buffer_with_data(data=np.array(texcoord, dtype='float32'), usage=wgpu.BufferUsage.VERTEX)
    self.ibo: wgpu.GPUBuffer = device.create_buffer_with_data(data=np.array(index, dtype='uint32'), usage=wgpu.BufferUsage.INDEX)

  def draw (self, st: State) -> None:
    first_instance = st.get_shader().commit_matrix(st)
    st.render_pass.set_vertex_buffer(0, self.coord_vbo)
    st.render_pass.set_vertex_buffer(1, self.texcoord_vbo)
    st.render_pass.set_index_buffer(self.ibo, wgpu.IndexFormat.uint32)
    st.render_pass.draw_indexed(self.nind, 1, 0, 0, first_instance)
