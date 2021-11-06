// Daniel Shiffman
// http://youtube.com/thecodingtrain
// https://thecodingtrain.com/CodingChallenges/111-animated-sprite.html

// Horse Spritesheet from
// https://opengameart.org/content/2d-platformer-art-assets-from-horse-of-spring

// Animated Sprite
// https://youtu.be/3noMeuufLZY

class Sprite {
  constructor(animation, x, y, speed) {
    this.x = x;
    this.y = y;
    this.r = 60;
    this.animation = animation;
    this.w = this.animation[0].width;
    this.len = this.animation.length;
    this.speed = speed;
    this.index = 0;
  }


  move() {
    //this.x -= 16;
    this.index += this.speed;
    //this.x += this.speed * 15;
    this.x -= this.speed * 15;

    
  }


  show() {
    let index = floor(this.index) % this.len;
    image(this.animation[index], this.x, this.y,this.r, this.r);
  }

  animate() {
    this.index += this.speed;
    //this.x += this.speed * 15;
    this.x -= this.speed * 15;

    /*if (this.x > width) {
      this.x = -this.w;
    }*/
    
    if (this.x <0){
    	this.x = width;
    }
    
    
  }
}
