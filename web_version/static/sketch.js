
//Based on Daniel Shiffman's
// https://thecodingtrain.com/CodingChallenges/147-chrome-dinosaur.html
// https://youtu.be/l0HoJHc-63Q
// Google Chrome Dinosaur Game (Unicorn, run!)
// https://editor.p5js.org/codingtrain/sketches/v3thq2uhk
let score = 0;
let high_score = 0;
let scroll = 10;
let scrollBg = 0;
let player;
let playerImg;
let sharkImg;
let bImg;
let sharks = [];
let animation = [];
let playeranimation=[];
let soundClassifier;
let finished;
let input;
let img;
let spritesheet;
let spritedata;
 

function preload() {
  const options = {
    probabilityThreshold: 0.95
  };
  sharkImg = loadImage('static/s1.png');
  bImg = loadImage('static/background_inf.jpg');
  animation.push(loadImage('static/s0.png'));  animation.push(loadImage('static/s0.png'));  animation.push(loadImage('static/s0.png'));
  animation.push(loadImage('static/s1.png'));  animation.push(loadImage('static/s1.png'));  animation.push(loadImage('static/s1.png'));
  animation.push(loadImage('static/s0.png'));  animation.push(loadImage('static/s0.png'));  animation.push(loadImage('static/s0.png'));
  animation.push(loadImage('static/s1.png'));  animation.push(loadImage('static/s1.png'));  animation.push(loadImage('static/s1.png'));
  animation.push(loadImage('static/s3.png'));  animation.push(loadImage('static/s3.png'));  animation.push(loadImage('static/s3.png'));
  animation.push(loadImage('static/s3.png'));  animation.push(loadImage('static/s3.png'));  animation.push(loadImage('static/s3.png'));
  
  spritedata = loadJSON('static/frames.json');
  spritesheet = loadImage('static/sprites.png')

}

function mousePressed() {
  sharks.push(new Sprite(animation, 640,random(370,400),random(0.3,0.55)));
}

function setup() {
  c = createCanvas(800, 450);

  let frames = spritedata.frames;
  for (let i=0;i<frames.length;i++){
    let pos = frames[i].position;
    console.log(pos);
    let img = spritesheet.get(pos.x,pos.y,pos.w,pos.h);
    playeranimation.push(img);
  }

  player = new PlayerSprite(playeranimation);
}


function keyPressed() {
  if (key == ' ') {
    player.jump();
  }
  if (finished){
    finished = false;
    sharks = [];
    score = 0;
    scollBg = 0;
    scroll = 10;
    loop();
  }
  if (key == 's' || key == 'S'){
  saveCanvas(c,"screenshot.png");
  }
}

function draw() {

  image(bImg, -scrollBg, 0, width,height);
  image(bImg, -scrollBg + width, 0,width,height);
  
  if (scrollBg > width) {
    scrollBg = 0;
  }

  if (random(1) < 0.6667 && frameCount % 70 == 0) {
    sharks.push(new Sprite(animation, 640,random(370,400),random(0.3,0.55)));
  }
  if (frameCount % 5 == 0) {
    score++;
  }
  if (score > high_score) {
        high_score = (score);}

  
  textSize(32);
  fill(255,128,128);
  text(`Score: ${score}`, 10, 30);
    
   fill(255,128,128);
  text(`High Score: ${high_score}`, 550, 30);

  for (let t of sharks) {
    t.move();
    t.show();
    if (player.hits(t)) {

      textSize(32);
      fill(0, 102, 153);
      text('Game Over 😢', 300, 200);
      text('Press anything to restart', 240, 240);
      text(`'s' or 'S' for a screenshot`, 260, 280);
      finished = true;
      noLoop();

    }
  }

  player.show();
  player.move();
  player.animate();
  scroll += 0.005;
  scrollBg += scroll / 5;

}


