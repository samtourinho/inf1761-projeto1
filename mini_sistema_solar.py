"""
INF1761 - Computacao Grafica - Projeto 1: Cena 2D com textura e animacao

Mini-sistema solar 2D (Sol, Mercurio, Terra e Lua) modelado como grafo de
cena, com texturas nos astros e no fundo.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

import wgpu
from rendercanvas.glfw import RenderCanvas, loop

# classes do grafo de cena (repositorio da disciplina, pasta irma)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "scene_graph-alunos", "python"))

from camera2d import Camera2D
from engine import Engine
from node import Node
from pipeline import Pipeline
from renderer import Renderer
from sampler import Sampler
from scene import Scene
from shader import Shader
from square import Square
from texture import Texture
from textureset import TextureSet
from transform import Transform

from disk import Disk

canvas: RenderCanvas
device: wgpu.GPUDevice
context: Any
renderer: Renderer
camera: Camera2D
scene: Scene
last_t: float = 0.0


class Gira (Engine):
  def __init__ (self, trf: Transform, vel: float) -> None:
    self.trf = trf
    self.vel = vel  # graus por segundo

  def update (self, dt: float) -> None:
    self.trf.rotate(self.vel * dt, 0, 0, 1)


def textura (device: wgpu.GPUDevice, sampler: Sampler, arquivo: str) -> TextureSet:
  tex = Texture(device, "decal_texture", os.path.join(_HERE, "Images", arquivo))
  return TextureSet([tex, sampler])


def initialize (device: wgpu.GPUDevice, target_format: str) -> None:
  global camera, scene
  camera = Camera2D(-3.5, 3.5, -3.5, 3.5)

  disk = Disk(device)
  sampler = Sampler(device, "decal_sampler")
  fundo_tex = textura(device, sampler, "space.jpg")
  sol_tex = textura(device, sampler, "sun.png")
  merc_tex = textura(device, sampler, "mercury.png")
  terra_tex = textura(device, sampler, "Terra.png")
  lua_tex = textura(device, sampler, "moon.png")

  # fundo
  trf = Transform(); trf.scale(8, 5, 1)
  fundo = Node(trf=trf, apps=[fundo_tex], shps=[Square(device)])

  # sol
  trf = Transform(); trf.scale(0.8, 0.8, 1)
  sol = Node(trf=trf, apps=[sol_tex], shps=[disk])

  # mercurio: translacao em torno do sol
  merc_trf = Transform()
  trf = Transform(); trf.translate(1.5, 0, 0); trf.scale(0.18, 0.18, 1)
  mercurio = Node(trf=merc_trf, nodes=[Node(trf=trf, apps=[merc_tex], shps=[disk])])

  # lua: translacao em torno da terra
  lua_trf = Transform()
  trf = Transform(); trf.translate(0.6, 0, 0); trf.scale(0.12, 0.12, 1)
  lua = Node(trf=lua_trf, nodes=[Node(trf=trf, apps=[lua_tex], shps=[disk])])

  # terra: translacao em torno do sol e rotacao propria
  # a lua fica fora do ramo da rotacao, para nao herdar o giro
  terra_trf = Transform()
  terra_rot = Transform()
  trf = Transform(); trf.scale(0.3, 0.3, 1)
  terra_corpo = Node(trf=terra_rot, nodes=[Node(trf=trf, apps=[terra_tex], shps=[disk])])
  trf = Transform(); trf.translate(2.5, 0, 0)
  terra = Node(trf=terra_trf, nodes=[Node(trf=trf, nodes=[terra_corpo, lua])])

  shader = Shader(device, os.path.join(_HERE, "shaders", "textured2d.wgsl"))
  shader.set_vertex_buffers([
    {"array_stride": 2 * 4, "step_mode": "vertex",
     "attributes": [{"format": "float32x2", "offset": 0, "var_name": "pos"}]},
    {"array_stride": 2 * 4, "step_mode": "vertex",
     "attributes": [{"format": "float32x2", "offset": 0, "var_name": "texcoord"}]},
  ])
  pipeline = Pipeline(shader, target_format, depth_stencil=None)
  for ts in [fundo_tex, sol_tex, merc_tex, terra_tex, lua_tex]:
    shader.add_texture_set(ts)

  # sem depth test: a ordem dos filhos e a ordem de desenho (fundo primeiro)
  root = Node(pipeline, nodes=[fundo, sol, mercurio, terra])
  scene = Scene(root)
  scene.add_engine(Gira(merc_trf, 65))
  scene.add_engine(Gira(terra_trf, 30))
  scene.add_engine(Gira(terra_rot, 250))
  scene.add_engine(Gira(lua_trf, 180))


def update (dt: float) -> None:
  scene.update(dt)


def draw () -> None:
  global last_t
  t = time.perf_counter()
  update(t - last_t)
  last_t = t

  target_texture = context.get_current_texture()
  renderer.render(target_texture, scene, camera)


def on_key (event: Any) -> None:
  if event["key"] == "q":
    canvas.close()


def main () -> None:
  global canvas, device, context, renderer, last_t

  canvas = RenderCanvas(size=(700, 700), title="Mini-sistema solar", update_mode="continuous", max_fps=60)
  adapter = wgpu.gpu.request_adapter_sync()
  device = adapter.request_device_sync()
  context = canvas.get_context("wgpu")
  target_format = context.get_preferred_format(device.adapter)
  context.configure(device=device, format=target_format)

  renderer = Renderer(device, clear_value=(0.0, 0.0, 0.0, 1.0))

  initialize(device, target_format)

  canvas.add_event_handler(on_key, "key_down")
  last_t = time.perf_counter()
  canvas.request_draw(draw)
  loop.run()


if __name__ == "__main__":
  main()
