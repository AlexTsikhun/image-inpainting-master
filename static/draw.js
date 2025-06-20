var canvas = document.getElementById("drawing");
const brushSizeInput = document.getElementById("brush-size");
const brushValue = document.getElementById("brush-size-value");

brushSizeInput.addEventListener("input", () => {
  const userValue = Math.round(brushSizeInput.value / 5) * 5;
  const actualBrushSize = userValue / 4;

  brushValue.textContent = userValue;
  brush.ctx.lineWidth = actualBrushSize;
});

function clear_canvas() {
  context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
}

var requestAnimationFrame =
  window.requestAnimationFrame ||
  window.mozRequestAnimationFrame ||
  window.webkitRequestAnimationFrame ||
  window.msRequestAnimationFrame;

/**
 * @param {CanvasRenderingContext2D} context
 */
function PrimitiveBrush(context) {
  if (!(context instanceof CanvasRenderingContext2D)) {
    throw new Error("No 2D rendering context given!");
  }

  this.ctx = context;
  this.strokes = [];
  this.strokeIndex = 0;
  this.workingStrokes;
  this.lastLength = 0;
  this.isTouching = false;

  // init context
  this.ctx.strokeStyle = "#FFF";

  this.ctx.lineWidth = parseInt(brushSizeInput.value); // Set initial brush size
  this.ctx.lineCap = this.ctx.lineJoin = "round";
}

brushSizeInput.addEventListener("input", function () {
  brush.ctx.lineWidth = parseInt(this.value);
});
/**
 * Begins a new stroke
 * @param  {MouseEvent} event
 */
PrimitiveBrush.prototype.start = function (event) {
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;

  this.workingStrokes = [{ x, y }];
  this.strokes.push(this.workingStrokes);
  this.lastLength = 1;
  this.isTouching = true;
  requestAnimationFrame(this._draw.bind(this));
};

/**
 * Moves the current position of our brush
 * @param  {MouseEvent} event
 */
PrimitiveBrush.prototype.move = function (event) {
  if (!this.isTouching) return;

  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;

  this.workingStrokes.push({
    x: x,
    y: y,
  });
  requestAnimationFrame(this._draw.bind(this));
};

/**
 * Stops a stroke
 * @param  {MouseEvent} event
 */
PrimitiveBrush.prototype.end = function (event) {
  if (this.workingStrokes && this.workingStrokes.length === 1) {
    const pt = this.workingStrokes[0];
    this.ctx.beginPath();
    this.ctx.arc(pt.x, pt.y, this.ctx.lineWidth / 2, 0, 2 * Math.PI);
    this.ctx.fill();
  }
  this.isTouching = false;
};

PrimitiveBrush.prototype._draw = function () {
  // save the current length quickly (it's dynamic)
  var length = this.workingStrokes.length;

  // return if there's no work to do
  if (length <= this.lastLength) {
    return;
  }

  var startIndex = this.lastLength - 1;

  this.lastLength = length;

  var pt0 = this.workingStrokes[startIndex];

  this.ctx.beginPath();

  this.ctx.moveTo(pt0.x, pt0.y);

  for (var j = startIndex; j < this.lastLength; j++) {
    var pt = this.workingStrokes[j];

    this.ctx.lineTo(pt.x, pt.y);
  }

  this.ctx.stroke();
};

var brush = new PrimitiveBrush(canvas.getContext("2d"));

canvas.addEventListener("mousedown", brush.start.bind(brush));
canvas.addEventListener("mousemove", brush.move.bind(brush));
canvas.addEventListener("mouseup", brush.end.bind(brush));
canvas.addEventListener("mouseout", brush.end.bind(brush));
