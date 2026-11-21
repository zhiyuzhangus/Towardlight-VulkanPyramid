"use strict";

const canvas = document.getElementById("scene");
const ctx = canvas.getContext("2d");
const byId = id => document.getElementById(id);
const clamp = (value, min, max) => Math.max(min, Math.min(value, max));

// Vertex coordinates, face order, and colors follow shaders/triangle.vert.
const vertices = [[0, .9, 0], [-.7, -.7, -.7], [.7, -.7, -.7], [.7, -.7, .7], [-.7, -.7, .7]];
const faces = [
  { indices: [0, 1, 2], color: "#ff402e" },
  { indices: [0, 2, 3], color: "#33ccf2" },
  { indices: [0, 3, 4], color: "#f2bf33" },
  { indices: [0, 4, 1], color: "#8c59f2" },
  { indices: [1, 4, 3, 2], color: "#33cc66" }
];
const home = { yaw: .66, pitch: .48, distance: 3.6, panX: 0, panY: 0, perspective: true };
const camera = { ...home };
const activePointers = new Map();
const minDistance = 2.2;
const maxDistance = 12;

function render() {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (!width || !height) return;
  const ratio = Math.min(window.devicePixelRatio || 1, 3);
  if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
  }
  ctx.setTransform(canvas.width / width, 0, 0, canvas.height / height, 0, 0);
  ctx.clearRect(0, 0, width, height);
  const cy = Math.cos(camera.yaw), sy = Math.sin(camera.yaw);
  const cp = Math.cos(camera.pitch), sp = Math.sin(camera.pitch);
  const focal = Math.min(width, height) / (2 * Math.tan(Math.PI / 8));
  const projected = vertices.map(([x, y, z]) => {
    const horizontal = sy * x + cy * z;
    const depth = camera.distance - (cp * horizontal + sp * y);
    const scale = focal / (camera.perspective ? depth : camera.distance);
    return [
      width * (.5 + camera.panX) + (cy * x - sy * z) * scale,
      height * (.48 + camera.panY) - (cp * y - sp * horizontal) * scale
    ];
  });
  for (const face of faces) {
    const points = face.indices.map(index => projected[index]);
    const [a, b, c] = points;
    const winding = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    if (winding <= 0) continue;
    ctx.fillStyle = face.color;
    ctx.beginPath();
    ctx.moveTo(...a);
    for (const point of points.slice(1)) ctx.lineTo(...point);
    ctx.closePath();
    ctx.fill();
  }
  byId("zoom-value").value = Math.round(100 * home.distance / camera.distance) + "%";
  byId("zoom-in").disabled = camera.distance <= minDistance;
  byId("zoom-out").disabled = camera.distance >= maxDistance;
  byId("perspective").setAttribute("aria-pressed", String(camera.perspective));
  byId("ortho").setAttribute("aria-pressed", String(!camera.perspective));
}

function orbit(dx, dy) {
  camera.yaw -= dx * .009;
  camera.pitch = clamp(camera.pitch + dy * .009, -1.55, 1.55);
}
function pan(dx, dy) {
  camera.panX += dx / canvas.clientWidth;
  camera.panY += dy / canvas.clientHeight;
}
function zoom(factor) {
  camera.distance = clamp(camera.distance * factor, minDistance, maxDistance);
}
function gesture() {
  const [a, b] = activePointers.values();
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, span: Math.hypot(a.x - b.x, a.y - b.y) };
}
canvas.addEventListener("pointerdown", event => {
  if (activePointers.size >= 2) return;
  canvas.focus({ preventScroll: true });
  canvas.setPointerCapture(event.pointerId);
  activePointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
});
canvas.addEventListener("pointermove", event => {
  const prev = activePointers.get(event.pointerId);
  if (!prev) return;
  const oldGesture = activePointers.size === 2 ? gesture() : null;
  activePointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
  if (oldGesture) {
    const next = gesture();
    if (next.span > 1 && oldGesture.span > 1) zoom(oldGesture.span / next.span);
    pan(next.x - oldGesture.x, next.y - oldGesture.y);
  } else {
    const dx = event.clientX - prev.x, dy = event.clientY - prev.y;
    if (event.shiftKey || event.ctrlKey || (event.buttons & 6)) pan(dx, dy);
    else orbit(dx, dy);
  }
  render();
});
for (const event of ["pointerup", "pointercancel", "lostpointercapture"]) {
  canvas.addEventListener(event, e => activePointers.delete(e.pointerId));
}
canvas.addEventListener("contextmenu", e => e.preventDefault());
canvas.addEventListener("wheel", event => {
  event.preventDefault();
  const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? canvas.clientHeight : 1;
  zoom(Math.exp(clamp(event.deltaY * unit * .0015, -1, 1)));
  render();
}, { passive: false });

const actions = {
  "zoom-in": () => zoom(1 / 1.15), "zoom-out": () => zoom(1.15),
  perspective: () => { camera.perspective = true; },
  ortho: () => { camera.perspective = false; },
  reset: () => Object.assign(camera, home)
};
for (const [id, action] of Object.entries(actions)) {
  byId(id).addEventListener("click", () => { action(); render(); });
}
canvas.addEventListener("keydown", event => {
  if (event.ctrlKey || event.metaKey || event.altKey) return;
  const move = event.shiftKey ? pan : orbit;
  const keys = {
    ArrowLeft: () => move(-12, 0), ArrowRight: () => move(12, 0),
    ArrowUp: () => move(0, -12), ArrowDown: () => move(0, 12),
    "+": actions["zoom-in"], "=": actions["zoom-in"], "-": actions["zoom-out"],
    r: actions.reset, p: actions.perspective, o: actions.ortho
  };
  const action = keys[event.key] || keys[event.key.toLowerCase()];
  if (!action) return;
  event.preventDefault();
  action();
  render();
});

if (matchMedia("(pointer: coarse)").matches) {
  byId("instructions").textContent = "Drag to orbit · Use two fingers to pan or pinch to zoom";
}
new ResizeObserver(render).observe(canvas);
render();
