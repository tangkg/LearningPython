import random
import pygame
from pygame.draw import rect
from pygame.locals import *
import time


class HeroPlane(pygame.sprite.Sprite):  # 精灵类 玩家和敌方都是sprite

    bullets = pygame.sprite.Group()  # 存放玩家飞机所有子弹的组

    def __init__(self, screen):
        pygame.sprite.Sprite.__init__(self)

        self.image = pygame.image.load("./image/plane_2.png")  # 玩家的飞机 https://www.shejihz.com/archives/84716/

        self.rect = self.image.get_rect()  # 根据图片获取矩形对象
        self.rect.topleft = [manager.bg_size[0] - 125, 680]  # 获取矩形左上角坐标

        self.speed = 10  # 飞机速度
        self.screen = screen  # 记录当前的窗口对象
        self.bullets = pygame.sprite.Group()  # 用来装sprite的，类似于列表

    def key_control(self):
        # 监听键盘事件
        key_pressed = pygame.key.get_pressed()

        if key_pressed[K_z] or key_pressed[K_UP]:
            self.rect.top -= self.speed
        if key_pressed[K_q] or key_pressed[K_LEFT]:
            self.rect.left -= self.speed
        if key_pressed[K_d] or key_pressed[K_RIGHT]:
            self.rect.right += self.speed
        if key_pressed[K_s] or key_pressed[K_DOWN]:
            self.rect.bottom += self.speed
        if key_pressed[K_SPACE]:  # 发射子弹
            bullet = Bullet(self.screen, self.rect.left, self.rect.top)
            # 把子弹放到列表中
            self.bullets.add(bullet)
            # 存放所有飞机子弹的组
            HeroPlane.bullets.add(bullet)

    def update(self):
        self.key_control()
        self.display()

    def display(self):  # 显示玩家飞机和所有子弹
        self.screen.blit(self.image, self.rect)
        self.bullets.update()  # 更新子弹坐标

        self.bullets.draw(self.screen)  # 把所有子弹添加到屏幕，相当于for遍历

    @classmethod
    def clear_bullets(cls):  # 清空子弹
        cls.bullets.empty()


class Bullet(pygame.sprite.Sprite):
    def __init__(self, screen, x, y):
        super().__init__()

        self.image = pygame.image.load("./image/bullet_2.png")  # http://47.100.44.116/sucai/vn2i05k0v.html

        self.rect = self.image.get_rect()  # 获取矩形对象
        self.rect.topleft = [x + 25, y - 45]
        self.screen = screen
        self.speed = 10  # 子弹速度

    def update(self):
        self.rect.top -= self.speed  # 修改子弹坐标
        # 如果子弹移出屏幕，则销毁子弹对象
        if self.rect.top <= -20:
            self.kill()  # 销毁


class EnemyPlane(pygame.sprite.Sprite):
    # 敌方所有子弹
    enemy_bullets = pygame.sprite.Group()

    def __init__(self, screen):
        super().__init__()

        # pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load("image/EnemyPlane_1.png")  # 50*53
        # 敌方飞机 https://img95.699pic.com/xsj/17/a7/9f.jpg!/fw/700/watermark/url/L3hzai93YXRlcl9kZXRhaWwyLnBuZw/align/southeast

        self.rect = self.image.get_rect()

        x = random.randrange(1, Manager.bg_size[0], 50)
        self.rect.topleft = [x, 0]

        self.speed = 4  # 飞机速度
        self.screen = screen  # 窗口对象
        self.bullets = pygame.sprite.Group()  # 装子弹到列表

        self.direction = 'right'  # 敌机移动方向

    def display(self):
        self.screen.blit(self.image, self.rect)
        self.bullets.update()  # 更新所有子弹
        self.bullets.draw(self.screen)

    def update(self):
        self.auto_move()
        self.auto_fire()
        self.display()

    def auto_move(self):
        if self.direction == 'right':  # 向右移动
            self.rect.right += self.speed
        elif self.direction == 'left':  # 向左移动
            self.rect.right -= self.speed

        if self.rect.right > Manager.bg_size[0] - 50:
            self.direction = 'left'
        elif self.rect.right <= 0:
            self.direction = 'right'

        self.rect.bottom += self.speed

    def auto_fire(self):  # 创建子弹对象，添加进列表
        random_num = random.randint(1, 30)
        if random_num == 8:  # 随便一个数字，用于缩概率
            bullet = EnemyBullet(self.screen, self.rect.left, self.rect.top)
            self.bullets.add(bullet)
            # 把敌方自动发的子弹添加到组里
            EnemyPlane.enemy_bullets.add(bullet)

    @classmethod  # 在确定重开是调用
    def clear_bullets(cls):  # 清空子弹
        cls.enemy_bullets.empty()


class EnemyBullet(pygame.sprite.Sprite):
    def __init__(self, screen, x, y):
        super().__init__()
        # pygame.sprite.Sprite.__init__(self)

        self.image = pygame.image.load("image/Enemy_bullet.png")  # https://www.shejihz.com/archives/148580/

        self.rect = self.image.get_rect()
        self.rect.topleft = [x + 21, y + 10]
        self.screen = screen
        self.speed = 8  # 子弹速度

    def display(self):  # 显示子弹到窗口上
        self.screen.blit(self.image, (self.rect.x, self.rect.y))

    def auto_move(self):  # 子弹飞
        self.rect.top += self.speed

    def update(self):  # 修改子弹坐标
        self.auto_move()  # 调用自动移动方法，更新子弹位置
        if self.rect.top > Manager.bg_size[1]:
            self.kill()


class GameSound(object):
    def __init__(self):
        pygame.mixer.init()  # 音乐模块初始化
        pygame.mixer.music.load('./bgm.mp3')
        pygame.mixer.music.set_volume(0.5)  # 声音大小

        self.bomb_sound = pygame.mixer.Sound(
            './bomb_1.mp3')  # https://img.tukuppt.com/newpreview_music/08/99/11/5c88e191c1bb717060.mp3

    def playBackgroundMusic(self):
        pygame.mixer.music.play(-1)  # -1无限循环播放音乐

    def playBombSound(self):
        pygame.mixer.Sound.play(self.bomb_sound)  # 爆炸音乐


class Bomb(object):
    def __init__(self, screen, type):
        self.screen = screen
        self.type = type
        if type == "enemy":  # 加载爆炸资源，用列表推导式
            self.mImage = [pygame.image.load("image/enemy0_down_" + str(v) + ".png") for v in range(1, 5)]
        else:
            self.mImage = [pygame.image.load("image/hero_blowup_n" + str(v) + ".png") for v in range(1, 5)]

        self.mIndex = 0  # 设置当前爆炸播放索引
        self.mPos = [0, 0]  # 爆炸设置
        self.mVisible = False  # 是否可见

    def action(self, rect):  # 触发爆炸方法draw,sprite类都有一个rect

        self.mPos[0] = rect.left
        self.mPos[1] = rect.top  # 取爆炸的左上角坐标
        self.mVisible = True

    def draw(self):
        if not self.mVisible:
            return
        self.screen.blit(self.mImage[self.mIndex], (self.mPos[0], self.mPos[1]))
        self.mIndex += 1
        if self.mIndex >= len(self.mImage):  # 如果下标已经到最后，则代表爆炸结束
            self.mIndex = 0  # 下标重置
            self.mVisible = False


class Map(object):

    def __init__(self, screen):  # 初始化地图
        self.mImage1 = pygame.image.load("./image/map.jpg")
        self.mImage2 = pygame.image.load("./image/map.jpg")
        # 窗口
        self.screen = screen
        # 辅助移动地图
        self.y1 = 0
        self.y2 = -Manager.bg_size[1]

    def move(self):
        self.y1 += 2
        self.y2 += 2
        if self.y1 >= Manager.bg_size[1]:  # 第一张图片移到窗口底部
            self.y1 = 0  # 回到开始坐标
        if self.y2 >= 0:
            self.y2 = -Manager.bg_size[1]

    def draw(self):
        self.screen.blit(self.mImage1, (0, self.y1))
        self.screen.blit(self.mImage2, (0, self.y2))


class Manager(object):
    bg_size = (480, 800)  # 元组地图大小
    # 创建敌机定时器id
    creat_enemy_id = 10
    # 游戏结束倒计时的id
    game_over_id = 11  # 取1-32之间的数字
    # 游戏是否结束
    is_game_over = False
    # 倒计时时长 3s
    over_time = 3

    def __init__(self):
        pygame.init()  # pygame的初始化
        self.screen = pygame.display.set_mode(Manager.bg_size, 0, 32)  # 创建窗口
        self.map = Map(self.screen)

        self.players = pygame.sprite.Group()  # 创建一个玩家精灵的group
        self.enemies = pygame.sprite.Group()  # 初始化一个装敌机精灵的group
        self.player_bomb = Bomb(self.screen, 'player')  # 初始化一个玩家爆炸的对象,type用作爆炸类中的判断
        self.enemy_bomb = Bomb(self.screen, 'enemy')  # 初始化一个玩家爆炸的对象
        self.sound = GameSound()

    def exit(self):
        print("GameOver")
        pygame.quit()
        exit()  # 退出python程序

    def show_count_down(self):  # 游戏结束，倒计时后重新开始
        self.drawText("Game over %d" % Manager.over_time, 80, Manager.bg_size[1]/2 - 20,
                      textHeight=40, fontColor=[255, 0, 0])

    def game_over_timer(self):
        self.show_count_down()
        Manager.over_time -= 1
        if Manager.over_time == 0:
            # 倒计时变为0，定时停止
            pygame.time.set_timer(Manager.game_over_id, 0)
            # 倒计时重新设置为3
            Manager.over_time = 3
            Manager.is_game_over = False
            self.start_game()

    def start_game(self):
        # 重新开始游戏，把所有飞机子弹类型清空
        EnemyPlane.clear_bullets()
        HeroPlane.clear_bullets()
        manager = Manager()  # 重新创建一个manager对象
        manager.main()  # 重新执行manager中的main方法

    def new_player(self):  # 创建玩家飞机对象，添加到玩家组
        player = HeroPlane(self.screen)
        self.players.add(player)

    def new_enemy(self):  # 创建敌方飞机对象，添加到敌机组
        enemy = EnemyPlane(self.screen)
        self.enemies.add(enemy)

    def drawText(self, text, x, y, textHeight=20, fontColor=(255, 0, 0), backgroundColor=None):
        # 分别是文字内容，坐标，文字大小，字体颜色，背景色None透明
        # 通过字体文件获取字体对象
        font_obj = pygame.font.Font('./ziti_1.ttf', textHeight)
        # 配置要显示的文件 用上一行生成的字体对象生成一个文字对象
        text_obj = font_obj.render(text, True, fontColor, backgroundColor)
        text_rect = text_obj.get_rect()  # 为文字获取一个矩形区域
        text_rect.topleft = (x, y)  # 设置显示对象的坐标
        # 绘制字到指定区域
        self.screen.blit(text_obj, text_rect)

    def main(self):
        self.sound.playBackgroundMusic()  # 放bgm
        self.new_player()  # 创建一个玩家对象
        # 开启创建敌机的定时器
        pygame.time.set_timer(Manager.creat_enemy_id, 1000)
        self.new_enemy()  # 创建一个敌机对象

        while True:

            # 移动地图并把地图贴到窗口上
            self.map.move()
            self.map.draw()
            # 绘制文字
            self.drawText('hp:10000', 0, 0)

            if Manager.is_game_over:
                self.show_count_down()

            for event in pygame.event.get():  # 判断事件类型
                if event.type == pygame.QUIT:
                    self.exit()
                elif event.type == Manager.creat_enemy_id:
                    # 创建一个敌机对象
                    self.new_enemy()
                elif event.type == Manager.game_over_id:
                    # 定时器触发事件
                    self.game_over_timer()

            # 调用爆炸的对象
            self.player_bomb.draw()
            self.enemy_bomb.draw()

            # 玩家飞机和敌机子弹的判断
            if self.players.sprites():
                is_over = pygame.sprite.spritecollide(self.players.sprites()[0], EnemyPlane.enemy_bullets, True)
                if is_over:
                    Manager.is_game_over = True
                    pygame.time.set_timer(Manager.game_over_id, 1000)  # 开始倒计时,1000ms启动间隔时间

                    self.player_bomb.action(self.players.sprites()[0].rect)
                    # 把玩家飞机从精灵组中移除
                    self.players.remove(self.players.sprites()[0])
                    self.sound.playBombSound()  # 爆炸声

            # 判断碰撞
            is_collide = pygame.sprite.groupcollide(self.players, self.enemies, True, True)
            # 2个True表示如果发生碰撞，则把两个飞机从页面上移除
            # 这一行代码最后返回一个字典，若不碰撞，则返回空字典一个
            if is_collide:  # 爆炸时

                Manager.is_game_over = True  # 标志游戏结束
                pygame.time.set_timer(Manager.game_over_id, 1000)  # 开启游戏倒计时

                items = list(is_collide.items())[0]
                # items获取碰撞的两架飞机，放到一个列表中

                x = items[0]  # 玩家飞机
                y = items[1][0]  # 敌机
                self.player_bomb.action(x.rect)  # 玩家爆炸图
                self.enemy_bomb.action(y.rect)  # 敌机爆炸图
                self.sound.playBombSound()  # 爆炸声音
            # 玩家子弹和所有敌机的碰撞判断
            is_enemy = pygame.sprite.groupcollide(HeroPlane.bullets, self.enemies, True, True)
            # 把玩家发的子弹和敌方飞机这两个对象传进来
            if is_enemy:
                items = list(is_enemy.items())[0]  # 生成字典一个
                y = items[1][0]
                # 敌机爆炸图片
                self.enemy_bomb.action(y.rect)
                # 敌机爆炸声音
                self.sound.playBombSound()

            self.players.update()  # 玩家飞机和子弹的显示
            self.enemies.update()  # 敌方飞机和子弹的显示

            pygame.display.update()  # 刷新窗口内容
            time.sleep(0.01)


if __name__ == '__main__':
    manager = Manager()
    manager.main()
