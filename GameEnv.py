import pygame
import math
from Walls import getWalls
from Goals import getGoals
import numpy as np

GOALREWARD = 1
LIFE_REWARD = 0
PENALTY = -0.5

def distance(pt1, pt2):
    return(((pt1.x - pt2.x)**2 + (pt1.y - pt2.y)**2)**0.5)

# Rotate point around origin and origin is the center of the car so it rotates around the center of the car
def rotate(origin,point,angle):
    qx = origin.x + math.cos(angle) * (point.x - origin.x) - math.sin(angle) * (point.y - origin.y)
    qy = origin.y + math.sin(angle) * (point.x - origin.x) + math.cos(angle) * (point.y - origin.y)
    q = myPoint(qx, qy)
    return q

# Rotate rectangle around center so When the car turns, instead of just moving it sideways, you rotate its rectangle corners around its center to simulate realistic turning
def rotateRect(pt1, pt2, pt3, pt4, angle):

    pt_center = myPoint((pt1.x + pt3.x)/2, (pt1.y + pt3.y)/2)

    pt1 = rotate(pt_center,pt1,angle)
    pt2 = rotate(pt_center,pt2,angle)
    pt3 = rotate(pt_center,pt3,angle)
    pt4 = rotate(pt_center,pt4,angle)

    return pt1, pt2, pt3, pt4

class myPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
class myLine:
    def __init__(self, pt1, pt2):
        self.pt1 = myPoint(pt1.x, pt1.y)
        self.pt2 = myPoint(pt2.x, pt2.y)

class Ray:
    def __init__(self,x,y,angle):
        self.x = x
        self.y = y
        self.angle = angle

    def cast(self, wall):
        x1 = wall.x1 
        y1 = wall.y1
        x2 = wall.x2
        y2 = wall.y2

        vec = rotate(myPoint(0,0), myPoint(0,-1000), self.angle)
        
        x3 = self.x
        y3 = self.y
        x4 = self.x + vec.x
        y4 = self.y + vec.y

        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            
        if den == 0:
            return None
        else:
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
            u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

            if t > 0 and t < 1 and u < 1 and u > 0:
                pt = myPoint(math.floor(x1 + t * (x2 - x1)), math.floor(y1 + t * (y2 - y1)))
                return(pt)

class Car:
    def __init__(self, x, y):
        self.pt = myPoint(x, y)
        self.x = x
        self.y = y
        self.width = 14
        self.height = 30

        self.points = 0

        self.original_image = pygame.image.load("car.png").convert()
        self.image = self.original_image  # This will reference the rotated image.
        self.image.set_colorkey((0,0,0))
        self.rect = self.image.get_rect().move(self.x, self.y)

        self.angle = math.radians(180)
        self.soll_angle = self.angle

        self.dvel = 1
        self.vel = 0
        self.velX = 0
        self.velY = 0
        self.maxvel = 15 # before 15

        self.angle = math.radians(180)
        self.soll_angle = self.angle

        self.pt1 = myPoint(self.pt.x - self.width / 2, self.pt.y - self.height / 2)
        self.pt2 = myPoint(self.pt.x + self.width / 2, self.pt.y - self.height / 2)
        self.pt3 = myPoint(self.pt.x + self.width / 2, self.pt.y + self.height / 2)
        self.pt4 = myPoint(self.pt.x - self.width / 2, self.pt.y + self.height / 2)

        self.p1 = self.pt1
        self.p2 = self.pt2
        self.p3 = self.pt3
        self.p4 = self.pt4

        self.distances = []
    

    def action(self, choice):
        if choice == 0:
            pass
        elif choice == 1: # accelerate
            self.accelerate(self.dvel) 
        elif choice == 8: # accelerate and turn right
            self.accelerate(self.dvel)
            self.turn(1)
        elif choice == 7: # accelerate and turn left
            self.accelerate(self.dvel)
            self.turn(-1)
        elif choice == 4: # brake
            self.accelerate(-self.dvel)
        elif choice == 5: # brake and turn right
            self.accelerate(-self.dvel)
            self.turn(1)
        elif choice == 6: # brake and turn left
            self.accelerate(-self.dvel)
            self.turn(-1)
        elif choice == 3: # turn right
            self.turn(1)
        elif choice == 2: # turn left
            self.turn(-1)
        else:
            print(f"Invalid choice: {choice}")
    
    def accelerate(self,dvel):
        self.vel = self.vel + dvel
        # Ensure velocity doesn't exceed max or go below 0
        self.vel = max(0, min(self.vel, self.maxvel))
        
    def turn(self, dir):
        """ Rotate the car by the given direction (1 for right, -1 for left). """
        self.soll_angle = self.soll_angle + dir * math.radians(5)
        # Keep the angle within 0 to 2pi
        if self.soll_angle >= 2 * math.pi:
            self.soll_angle -= 2 * math.pi
        elif self.soll_angle < 0:
            self.soll_angle += 2 * math.pi
    
    def update(self):        
        self.angle = self.soll_angle

        # Handle zero velocity before performing rotation
        if self.vel != 0:
            vec_temp = rotate(myPoint(0, 0), myPoint(0, self.vel), self.angle)
            self.velX, self.velY = vec_temp.x, vec_temp.y
        else:
            self.velX = 0
            self.velY = 0  # No movement if velocity is zero

        self.x = self.x + self.velX
        self.y = self.y + self.velY

        self.rect.center = self.x, self.y

        self.pt1 = myPoint(self.pt1.x + self.velX, self.pt1.y + self.velY)
        self.pt2 = myPoint(self.pt2.x + self.velX, self.pt2.y + self.velY)
        self.pt3 = myPoint(self.pt3.x + self.velX, self.pt3.y + self.velY)
        self.pt4 = myPoint(self.pt4.x + self.velX, self.pt4.y + self.velY)

        # rotate the rectangle around its center
        self.p1 ,self.p2 ,self.p3 ,self.p4  = rotateRect(self.pt1, self.pt2, self.pt3, self.pt4, self.soll_angle)

        self.image = pygame.transform.rotate(self.original_image, 90 - self.soll_angle * 180 / math.pi)
        x, y = self.rect.center  # Save its current center.
        self.rect = self.image.get_rect()  # Replace old rect with new rect.
        self.rect.center = (x, y)

    def cast(self, walls):

        ray1 = Ray(self.x, self.y, self.soll_angle)
        ray2 = Ray(self.x, self.y, self.soll_angle - math.radians(30))
        ray3 = Ray(self.x, self.y, self.soll_angle + math.radians(30))
        ray4 = Ray(self.x, self.y, self.soll_angle + math.radians(45))
        ray5 = Ray(self.x, self.y, self.soll_angle - math.radians(45))
        ray6 = Ray(self.x, self.y, self.soll_angle + math.radians(90))
        ray7 = Ray(self.x, self.y, self.soll_angle - math.radians(90))
        ray8 = Ray(self.x, self.y, self.soll_angle + math.radians(180))
        ray9 = Ray(self.x, self.y, self.soll_angle + math.radians(10))
        ray10 = Ray(self.x, self.y, self.soll_angle - math.radians(10))
        ray11 = Ray(self.x, self.y, self.soll_angle + math.radians(135))
        ray12 = Ray(self.x, self.y, self.soll_angle - math.radians(135))
        ray13 = Ray(self.x, self.y, self.soll_angle + math.radians(20))
        ray14 = Ray(self.x, self.y, self.soll_angle - math.radians(20))
        ray15 = Ray(self.p1.x,self.p1.y, self.soll_angle + math.radians(90))
        ray16 = Ray(self.p2.x,self.p2.y, self.soll_angle - math.radians(90))
        ray17 = Ray(self.p1.x,self.p1.y, self.soll_angle + math.radians(0))
        ray18 = Ray(self.p2.x,self.p2.y, self.soll_angle - math.radians(0))

        self.rays = []
        self.rays.append(ray1)
        self.rays.append(ray2)
        self.rays.append(ray3)
        self.rays.append(ray4)
        self.rays.append(ray5)
        self.rays.append(ray6)
        self.rays.append(ray7)
        self.rays.append(ray8)
        self.rays.append(ray9)
        self.rays.append(ray10)
        self.rays.append(ray11)
        self.rays.append(ray12)
        self.rays.append(ray13)
        self.rays.append(ray14)
        self.rays.append(ray15)
        self.rays.append(ray16)
        self.rays.append(ray17)
        self.rays.append(ray18)


        observations = []
        self.closestRays = []

        for ray in self.rays:
            closest = None #myPoint(0,0)
            record = math.inf
            for wall in walls:
                pt = ray.cast(wall)
                if pt:
                    dist = distance(myPoint(self.x, self.y),pt)
                    if dist < record:
                        record = dist
                        closest = pt

            if closest: 
                #append distance for current ray 
                self.closestRays.append(closest)
                observations.append(record)
               
            else:
                observations.append(1000)

        for i in range(len(observations)):
            #invert observation values 0 is far away 1 is close
            # observations[i] = ((1000 - observations[i]) / 1000)
            observations[i] = max(0, min(1, (1000 - observations[i]) / 1000))  # Ensure it's in range [0, 1]

        observations.append(self.vel / self.maxvel)
        return observations

    def collision(self, wall):

        line1 = myLine(self.p1, self.p2)
        line2 = myLine(self.p2, self.p3)
        line3 = myLine(self.p3, self.p4)
        line4 = myLine(self.p4, self.p1)

        x1 = wall.x1 
        y1 = wall.y1
        x2 = wall.x2
        y2 = wall.y2

        lines = []
        lines.append(line1)
        lines.append(line2)
        lines.append(line3)
        lines.append(line4)

        for li in lines:
            
            x3 = li.pt1.x
            y3 = li.pt1.y
            x4 = li.pt2.x
            y4 = li.pt2.y

            den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            
            if(den == 0):
                continue
            else:
                t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
                u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

                if t > 0 and t < 1 and u < 1 and u > 0:
                    return(True)
        
        return(False)
    
    def score(self, goal):
        
        line1 = myLine(self.p1, self.p3)

        vec = rotate(myPoint(0,0), myPoint(0,-50), self.angle)
        line1 = myLine(myPoint(self.x,self.y),myPoint(self.x + vec.x, self.y + vec.y))

        x1 = goal.x1 
        y1 = goal.y1
        x2 = goal.x2
        y2 = goal.y2
            
        x3 = line1.pt1.x
        y3 = line1.pt1.y
        x4 = line1.pt2.x
        y4 = line1.pt2.y

        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if(den == 0):
            return 0
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

        if t > 0 and t < 1 and u < 1 and u > 0:
            pt = math.floor(x1 + t * (x2 - x1)), math.floor(y1 + t * (y2 - y1))

            d = distance(myPoint(self.x, self.y), myPoint(pt[0], pt[1]))
            if d < 20:
                self.points += GOALREWARD
                return GOALREWARD

        return 0

    def reset(self):

        self.x = 50
        self.y = 300
        self.velX = 0
        self.velY = 0
        self.vel = 0
        self.angle = math.radians(180)
        self.soll_angle = self.angle
        self.points = 0

        self.pt1 = myPoint(self.pt.x - self.width / 2, self.pt.y - self.height / 2)
        self.pt2 = myPoint(self.pt.x + self.width / 2, self.pt.y - self.height / 2)
        self.pt3 = myPoint(self.pt.x + self.width / 2, self.pt.y + self.height / 2)
        self.pt4 = myPoint(self.pt.x - self.width / 2, self.pt.y + self.height / 2)

        self.p1 = self.pt1
        self.p2 = self.pt2
        self.p3 = self.pt3
        self.p4 = self.pt4

    def draw(self, win):
        win.blit(self.image, self.rect)
  
class RacingEnv:

    def __init__(self):
        pygame.init()
        self.font = pygame.font.Font(pygame.font.get_default_font(), 36)

        self.fps = 120
        self.width = 1000
        self.height = 600
        self.history = []

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("RACING DQN")
        self.screen.fill((0,0,0))
        self.back_image = pygame.image.load("track.png").convert()
        self.back_rect = self.back_image.get_rect().move(0, 0)
        self.action_space = None
        self.observation_space = None
        self.game_reward = 0
        self.score = 0
 
        self.reset()


    def reset(self):
        self.screen.fill((0, 0, 0))

        self.car = Car(50, 300)
        self.walls = getWalls()
        self.goals = getGoals()
        self.game_reward = 0

    def step(self, action):

        done = False
        self.car.action(action)
        self.car.update()
        reward = LIFE_REWARD

        # Check if car passes Goal and scores
        index = 1
        for goal in self.goals:
            
            if index > len(self.goals):
                index = 1
            if goal.isactiv:
                if self.car.score(goal):
                    goal.isactiv = False
                    self.goals[index-2].isactiv = True
                    reward += GOALREWARD

            index = index + 1

        #check if car crashed in the wall
        for wall in self.walls:
            if self.car.collision(wall):
                reward += PENALTY
                done = True

        # --- Reward shaping: negative distance to next goal ---
        # Find the next active goal
        active_goal = None
        for goal in self.goals:
            if goal.isactiv:
                active_goal = goal
                break
        
        if active_goal is not None:
            # Use the center of the goal line as the target
            goal_center_x = (active_goal.x1 + active_goal.x2) / 2
            goal_center_y = (active_goal.y1 + active_goal.y2) / 2
            car_pos = myPoint(self.car.x, self.car.y)
            goal_pos = myPoint(goal_center_x, goal_center_y)
            dist = distance(car_pos, goal_pos)

            # Check for invalid distance or NaN
            if np.isnan(dist) or np.isinf(dist):
                print("❌ Invalid distance value detected!")
                dist = 0.0  # Reset invalid distance to zero

            # Normalize distance (optional, depending on your track size)
            norm_dist = dist / 1000.0  # adjust denominator as needed
            reward -= 0.1 * norm_dist  # small penalty for being far from goal

        # Reward or penalty for velocity
        if self.car.maxvel > 0:
            reward += 0.1 * (self.car.vel / self.car.maxvel)

        new_state = np.array(self.car.cast(self.walls), dtype=np.float32)

        # Ensure new_state is valid and not NaN
        if np.isnan(new_state).any() or np.isinf(new_state).any():
            print("❌ Invalid state detected!")
            new_state = np.zeros_like(new_state)  # Reset to zero or handle appropriately

        #normalize states
        if done:
            new_state = np.zeros_like(new_state)

        # Ensure reward is valid
        if np.isnan(reward) or np.isinf(reward):
            print("❌ Invalid reward detected!")
            reward = 0.0  # Reset reward to zero if invalid

        return new_state, reward, done

    def render(self, action):

        DRAW_WALLS = False
        DRAW_GOALS = True
        DRAW_RAYS = False

        pygame.time.delay(10)

        self.clock = pygame.time.Clock()
        self.screen.fill((0, 0, 0))

        self.screen.blit(self.back_image, self.back_rect)

        if DRAW_WALLS:
            for wall in self.walls:
                wall.draw(self.screen)
        
        if DRAW_GOALS:
            for goal in self.goals:
                goal.draw(self.screen)
                if goal.isactiv:
                    goal.draw(self.screen)
        
        self.car.draw(self.screen)

        if DRAW_RAYS:
            i = 0
            for pt in self.car.closestRays:
                pygame.draw.circle(self.screen, (0,0,255), (pt.x, pt.y), 5)
                i += 1
                if i < 15:
                    pygame.draw.line(self.screen, (255,255,255), (self.car.x, self.car.y), (pt.x, pt.y), 1)
                elif i >=15 and i < 17:
                    pygame.draw.line(self.screen, (255,255,255), ((self.car.p1.x + self.car.p2.x)/2, (self.car.p1.y + self.car.p2.y)/2), (pt.x, pt.y), 1)
                elif i == 17:
                    pygame.draw.line(self.screen, (255,255,255), (self.car.p1.x , self.car.p1.y ), (pt.x, pt.y), 1)
                else:
                    pygame.draw.line(self.screen, (255,255,255), (self.car.p2.x, self.car.p2.y), (pt.x, pt.y), 1)

        #render controll
        pygame.draw.rect(self.screen,(255,255,255),(800, 100, 40, 40),2)
        pygame.draw.rect(self.screen,(255,255,255),(850, 100, 40, 40),2)
        pygame.draw.rect(self.screen,(255,255,255),(900, 100, 40, 40),2)
        pygame.draw.rect(self.screen,(255,255,255),(850, 50, 40, 40),2)

        if action == 4:
            pygame.draw.rect(self.screen,(0,255,0),(850, 50, 40, 40)) 
        elif action == 6:
            pygame.draw.rect(self.screen,(0,255,0),(850, 50, 40, 40))
            pygame.draw.rect(self.screen,(0,255,0),(800, 100, 40, 40))
        elif action == 5:
            pygame.draw.rect(self.screen,(0,255,0),(850, 50, 40, 40))
            pygame.draw.rect(self.screen,(0,255,0),(900, 100, 40, 40))
        elif action == 1:
            pygame.draw.rect(self.screen,(0,255,0),(850, 100, 40, 40)) 
        elif action == 8:
            pygame.draw.rect(self.screen,(0,255,0),(850, 100, 40, 40))
            pygame.draw.rect(self.screen,(0,255,0),(800, 100, 40, 40))
        elif action == 7:
            pygame.draw.rect(self.screen,(0,255,0),(850, 100, 40, 40))
            pygame.draw.rect(self.screen,(0,255,0),(900, 100, 40, 40))
        elif action == 2:
            pygame.draw.rect(self.screen,(0,255,0),(800, 100, 40, 40))
        elif action == 3:
            pygame.draw.rect(self.screen,(0,255,0),(900, 100, 40, 40))

        # score
        text_surface = self.font.render(f'Points {self.car.points/2}', True, pygame.Color('green'))
        self.screen.blit(text_surface, dest=(0, 0))
        # speed
        text_surface = self.font.render(f'Velocity {self.car.vel}', True, pygame.Color('green'))
        self.screen.blit(text_surface, dest=(800, 0))

        self.clock.tick(self.fps)
        pygame.display.update()

    def close(self):
        pygame.quit()

