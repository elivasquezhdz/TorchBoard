// Daniel Shiffman
// https://thecodingtrain.com/CodingChallenges/147-chrome-dinosaur.html
// https://youtu.be/l0HoJHc-63Q

// Google Chrome Dinosaur Game (Unicorn, run!)
// https://editor.p5js.org/codingtrain/sketches/v3thq2uhk

class Player {
  constructor() {
    this.r = 100;
    this.x = 50;
    this.y = 200 ;//- this.r;
    this.vy = 0;
    this.gravity = 3;
  }

  jump() {
    if (this.y == height - 150) {
      this.vy = -40;
    }
  }

  hits(shark) {
    let x1 = this.x + this.r * 0.5;
    let y1 = this.y + this.r * 0.5;
    let x2 = shark.x + shark.r * 0.5;
    let y2 = shark.y + shark.r * 0.5;
    return collideCircleCircle(x1, y1, this.r, x2, y2, shark.r);
  }
    hits(sprite) {
    let x1 = this.x + this.r * 0.5;
    let y1 = this.y + this.r * 0.5;
    let x2 = sprite.x + sprite.r * 0.5;
    let y2 = sprite.y + sprite.r * 0.5;
    return collideCircleCircle(x1, y1, this.r, x2, y2, sprite.r);
  }


  move() {
    this.y += this.vy;
    this.vy += this.gravity;
    this.y = constrain(this.y, 0, height - 150);//this.r);
  }

  show() {
    //image(uImg, this.x, this.y, this.r, this.r);
    image(playerImg, this.x, this.y, this.r, 150);

    // fill(255, 50);
    // ellipseMode(CORNER);
    // ellipse(this.x, this.y, this.r, this.r);
  }
}
