#!/usr/bin/env python3
"""
PortraitPay Celebrity Face Database Builder
============================================
Downloads celebrity face images and pre-computes embeddings for face matching.

Categories:
- 🎬 Film (影视明星)
- ⚽ Sports (体育名人)
- 💼 Business (商界领袖)
- 📺 TV (电视主持人/综艺明星)

Note: Chinese mainland politicians are NOT included - they are for risk detection only.
"""

import os
import json
import hashlib
import time
import sqlite3
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "celebrity_data"
DB_PATH = SCRIPT_DIR / "data" / "portraitpay.db"
MODEL_DIR = SCRIPT_DIR / "models"
IMAGE_DIR = DATA_DIR / "images"

DATA_DIR.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# Political figures for RISK DETECTION
# Format: (name, category, subcategory, description, country, risk_level)
POLITICS = [
    # US Politicians - WARNING risk
    ("Joe Biden", "politics", "politics/current", "US President", "us", "warning"),
    ("Donald Trump", "politics", "politics/current", "US President (current term)", "us", "warning"),
    ("Barack Obama", "politics", "politics/former", "Former US President", "us", "warning"),
    ("Hillary Clinton", "politics", "politics/former", "Former US Secretary of State", "us", "warning"),
    ("Kamala Harris", "politics", "politics/current", "US Vice President", "us", "warning"),
    ("Nancy Pelosi", "politics", "politics/current", "US House Speaker", "us", "warning"),
    ("Ron DeSantis", "politics", "politics/current", "US Governor of Florida", "us", "warning"),
    ("Gavin Newsom", "politics", "politics/current", "US Governor of California", "us", "warning"),
    ("MTG", "politics", "politics/current", "US Congresswoman Marjorie Taylor Greene", "us", "warning"),
    ("Alexandria Ocasio-Cortez", "politics", "politics/current", "US Congresswoman AOC", "us", "warning"),
    ("Mike Pompeo", "politics", "politics/former", "Former US Secretary of State", "us", "warning"),
    ("Antony Blinken", "politics", "politics/current", "US Secretary of State", "us", "warning"),

    # European Leaders - WARNING risk
    ("马克龙", "politics", "politics/current", "法国总统", "fr", "warning"),
    ("朔尔茨", "politics", "politics/current", "德国总理", "de", "warning"),
    ("苏纳克", "politics", "politics/current", "英国首相", "uk", "warning"),
    ("梅洛尼", "politics", "politics/current", "意大利总理", "it", "warning"),
    ("冯德莱恩", "politics", "politics/current", "欧盟委员会主席", "eu", "warning"),
    ("米歇尔", "politics", "politics/current", "欧洲理事会主席", "eu", "warning"),
    ("约翰逊", "politics", "politics/former", "英国前首相", "uk", "warning"),
    ("默克尔", "politics", "politics/former", "德国前总理", "de", "warning"),

    # Asian Leaders - mixed risk
    ("习近平", "politics", "politics/blocked_cn", "中国国家主席", "cn", "blocked"),
    ("李克强", "politics", "politics/blocked_cn", "中国前总理", "cn", "blocked"),
    ("蔡英文", "politics", "politics/blocked_cn", "台湾地区领导人", "tw", "blocked"),
    ("尹锡悦", "politics", "politics/current", "韩国总统", "kr", "warning"),
    ("岸田文雄", "politics", "politics/current", "日本首相", "jp", "warning"),
    ("莫迪", "politics", "politics/current", "印度总理", "in", "warning"),
    ("李显龙", "politics", "politics/current", "新加坡总理", "sg", "warning"),
    ("巴育", "politics", "politics/former", "泰国前总理", "th", "warning"),

    # Middle East - WARNING/BLOCKED risk
    ("内塔尼亚胡", "politics", "politics/current", "以色列总理", "il", "warning"),
    ("普京", "politics", "politics/blocked", "俄罗斯总统", "ru", "blocked"),
    ("埃尔多安", "politics", "politics/current", "土耳其总统", "tr", "warning"),
    ("哈梅内伊", "politics", "politics/blocked", "伊朗最高领袖", "ir", "blocked"),
    ("阿卜杜勒·梅蒂", "politics", "politics/current", "沙特阿拉伯王储", "sa", "warning"),

    # Latin America - WARNING risk
    ("卢拉", "politics", "politics/current", "巴西总统", "br", "warning"),
    ("博索纳罗", "politics", "politics/former", "巴西前总统", "br", "warning"),
    ("马克里", "politics", "politics/former", "阿根廷前总统", "ar", "warning"),
]

# Celebrity database (well-known international and Chinese celebrities)
# Format: (name, category, subcategory, description, country)
# Subcategories: film/action_star, film/comedian, film/actress, film/director, film/legend
CELEBRITIES = [
    # 🎬 Film - Chinese Mainland / HK / TW
    ("周星驰", "film", "film/comedian", "喜剧之王导演/演员", "cn"),
    ("成龙", "film", "film/action_star", "功夫巨星", "cn"),
    ("刘德华", "film", "film/legend", "天王级演员/歌手", "hk"),
    ("周润发", "film", "film/legend", "赌神主演", "hk"),
    ("李连杰", "film", "film/action_star", "功夫皇帝", "cn"),
    ("甄子丹", "film", "film/action_star", "武术演员", "hk"),
    ("梁朝伟", "film", "film/legend", "文艺片巨星", "hk"),
    ("张国荣", "film", "film/legend", "传奇演员/歌手", "hk"),
    ("巩俐", "film", "film/actress", "国际影后", "cn"),
    ("章子怡", "film", "film/actress", "国际影星", "cn"),
    ("范冰冰", "film", "film/actress", "知名演员", "cn"),
    ("黄晓明", "film", "film/actor", "知名演员", "cn"),
    ("邓超", "film", "film/actor", "知名演员", "cn"),
    ("吴京", "film", "film/action_star", "战狼导演/演员", "cn"),
    ("沈腾", "film", "film/comedian", "喜剧演员", "cn"),
    ("马丽", "film", "film/actress", "喜剧演员", "cn"),
    ("贾玲", "film", "film/comedian", "导演/演员", "cn"),
    # Chinese Mainland / HK / TW - additional
    ("李小龙", "film", "film/legend", "功夫传奇", "hk"),
    ("侯佩岑", "film", "film/actress", "台湾主持人/演员", "tw"),
    ("周杰伦", "film", "film/actor", "歌手/演员", "tw"),
    ("王力宏", "film", "film/actor", "歌手/演员", "tw"),
    ("林志玲", "film", "film/actress", "台湾模特/演员", "tw"),
    ("赵又廷", "film", "film/actor", "台湾演员", "tw"),
    ("彭于晏", "film", "film/actor", "台湾演员", "tw"),
    ("胡歌", "film", "film/actor", "知名演员", "cn"),
    ("霍建华", "film", "film/actor", "知名演员", "cn"),
    ("陈道明", "film", "film/actor", "资深演员", "cn"),
    ("葛优", "film", "film/comedian", "喜剧演员", "cn"),
    ("陈凯歌", "film", "film/director", "著名导演", "cn"),
    ("张艺谋", "film", "film/director", "著名导演", "cn"),
    ("陈可辛", "film", "film/director", "著名导演", "hk"),
    ("王家卫", "film", "film/director", "著名导演", "hk"),
    ("贾樟柯", "film", "film/director", "著名导演", "cn"),
    ("冯小刚", "film", "film/director", "著名导演", "cn"),

    # 🎬 Film - Hollywood (US)
    ("Tom Cruise", "film", "film/action_star", "Mission Impossible star", "us"),
    ("Brad Pitt", "film", "film/actor", "Actor, Fight Club", "us"),
    ("Angelina Jolie", "film", "film/actress", "Actress, Maleficent", "us"),
    ("Leonardo DiCaprio", "film", "film/actor", "Actor, Titanic", "us"),
    ("Scarlett Johansson", "film", "film/actress", "Actress, Avengers", "us"),
    ("Robert Downey Jr", "film", "film/action_star", "Actor, Iron Man", "us"),
    ("Jennifer Lawrence", "film", "film/actress", "Actress, Hunger Games", "us"),
    ("Dwayne Johnson", "film", "film/action_star", "Wrestler/Actor, Fast & Furious", "us"),
    ("Will Smith", "film", "film/actor", "Actor, Bad Boys", "us"),
    ("Morgan Freeman", "film", "film/actor", "Actor, Shawshank Redemption", "us"),
    ("Sylvester Stallone", "film", "film/action_star", "Actor, Rocky", "us"),
    ("Arnold Schwarzenegger", "film", "film/action_star", "Actor, Terminator", "us"),
    ("Keanu Reeves", "film", "film/action_star", "Actor, Matrix", "us"),
    ("Johnny Depp", "film", "film/actor", "Actor, Pirates of Caribbean", "us"),
    ("Chris Evans", "film", "film/action_star", "Actor, Captain America", "us"),
    ("Chris Hemsworth", "film", "film/action_star", "Actor, Thor", "us"),
    ("Mark Wahlberg", "film", "film/actor", "Actor, Transformers", "us"),
    ("Tom Hanks", "film", "film/actor", "Actor, Forrest Gump", "us"),
    ("Meryl Streep", "film", "film/actress", "Actress, The Iron Lady", "us"),
    ("Kate Winslet", "film", "film/actress", "Actress, Titanic", "us"),
    ("Denzel Washington", "film", "film/actor", "Actor, Training Day", "us"),
    ("Ryan Gosling", "film", "film/actor", "Actor, La La Land", "us"),
    ("Emma Stone", "film", "film/actress", "Actress, La La Land", "us"),
    ("Margot Robbie", "film", "film/actress", "Actress, Barbie", "us"),
    ("Ryan Reynolds", "film", "film/actor", "Actor, Deadpool", "us"),
    ("Hugh Jackman", "film", "film/actor", "Actor, Wolverine", "us"),
    ("Christian Bale", "film", "film/actor", "Actor, The Dark Knight", "us"),
    ("Heath Ledger", "film", "film/actor", "Actor, The Dark Knight", "us"),

    # 🎬 Film - Korean
    ("马东锡", "film", "film/action_star", "韩国动作演员", "kr"),
    ("宋康昊", "film", "film/actor", "韩国影帝", "kr"),
    ("金秀贤", "film", "film/actor", "韩国演员", "kr"),
    ("全智贤", "film", "film/actress", "韩国演员", "kr"),
    ("裴斗娜", "film", "film/actress", "韩国演员", "kr"),
    ("孔刘", "film", "film/actor", "韩国演员", "kr"),
    ("裴勇俊", "film", "film/actor", "韩国演员", "kr"),
    ("李敏镐", "film", "film/actor", "韩国演员", "kr"),
    ("崔岷植", "film", "film/actor", "韩国演员", "kr"),
    ("黄政民", "film", "film/actor", "韩国演员", "kr"),
    ("河正宇", "film", "film/actor", "韩国演员", "kr"),
    ("郑雨盛", "film", "film/actor", "韩国演员", "kr"),
    ("姜东元", "film", "film/actor", "韩国演员", "kr"),
    ("林允儿", "film", "film/actress", "韩国演员", "kr"),
    ("崔雪莉", "film", "film/actress", "韩国演员", "kr"),
    ("刘亚仁", "film", "film/actor", "韩国演员", "kr"),

    # 🎬 Film - Japanese
    ("木村拓哉", "film", "film/actor", "日本演员", "jp"),
    ("藤原龙也", "film", "film/actor", "日本演员", "jp"),
    ("小松菜奈", "film", "film/actress", "日本演员", "jp"),
    ("桥本环奈", "film", "film/actress", "日本演员", "jp"),
    ("新垣结衣", "film", "film/actress", "日本演员", "jp"),
    ("山田孝之", "film", "film/actor", "日本演员", "jp"),
    ("妻夫木聪", "film", "film/actor", "日本演员", "jp"),
    ("洼田正孝", "film", "film/actor", "日本演员", "jp"),
    ("永野芽郁", "film", "film/actress", "日本演员", "jp"),
    ("佐藤健", "film", "film/actor", "日本演员", "jp"),
    ("三浦友和", "film", "film/actor", "日本演员", "jp"),
    ("宫崎骏", "film", "film/director", "动画电影导演", "jp"),
    ("北野武", "film", "film/director", "著名导演", "jp"),
    ("是枝裕和", "film", "film/director", "著名导演", "jp"),

    # 🎬 Film - Indian/Bollywood
    ("Amitabh Bachchan", "film", "film/legend", "印度影帝", "in"),
    ("Shah Rukh Khan", "film", "film/legend", "Bollywood King", "in"),
    ("Aamir Khan", "film", "film/actor", "Bollywood演员", "in"),
    ("Salman Khan", "film", "film/action_star", "Bollywood演员", "in"),
    ("Deepika Padukone", "film", "film/actress", "Bollywood演员", "in"),
    ("Priyanka Chopra", "film", "film/actress", "Bollywood演员", "in"),
    ("Ranbir Kapoor", "film", "film/actor", "Bollywood演员", "in"),
    ("Alia Bhatt", "film", "film/actress", "Bollywood演员", "in"),
    ("Aishwarya Rai", "film", "film/actress", "Bollywood演员", "in"),

    # 🎬 Film - European
    ("Jean Dujardin", "film", "film/actor", "法国演员", "fr"),
    ("Marion Cotillard", "film", "film/actress", "法国演员", "fr"),
    ("Daniel Day-Lewis", "film", "film/actor", "英国演员", "gb"),
    ("Michael Caine", "film", "film/legend", "英国演员", "gb"),
    ("Helen Mirren", "film", "film/actress", "英国演员", "gb"),
    ("Eddie Redmayne", "film", "film/actor", "英国演员", "gb"),
    ("Romy Schneider", "film", "film/actress", "德国演员", "de"),
    ("Bruno Ganz", "film", "film/actor", "瑞士演员", "de"),

    # ⚽ Sports - Football
    ("梅西", "sports", "football", "阿根廷球王, FIFA最佳球员", "ar"),
    ("C罗", "sports", "football", "葡萄牙球星, 5次金球奖", "pt"),
    ("姆巴佩", "sports", "football", "法国球星, 世界杯冠军", "fr"),
    ("内马尔", "sports", "football", "巴西球星", "br"),
    ("哈兰德", "sports", "football", "挪威球星, 曼城前锋", "no"),
    ("德布劳内", "sports", "football", "比利时中场大师, 曼城核心", "be"),
    ("莫德里奇", "sports", "football", "克罗地亚中场, 皇马传奇", "hr"),
    ("贝利", "sports", "football", "巴西球王, 三届世界杯冠军", "br"),
    ("马拉多纳", "sports", "football", "阿根廷球王", "ar"),
    ("罗纳尔多", "sports", "football", "巴西传奇球星", "br"),
    ("齐达内", "sports", "football", "法国传奇中场/教练", "fr"),
    ("贝克汉姆", "sports", "football", "英格兰传奇球星", "gb"),
    ("久保建英", "sports", "football", "日本球星, 皇家社会", "jp"),
    ("三笘薰", "sports", "football", "日本球星, 布莱顿", "jp"),
    ("武磊", "sports", "football", "中国球星", "cn"),
    ("孙兴慜", "sports", "football", "韩国球星, 热刺", "kr"),

    # ⚽ Sports - Basketball
    ("姚明", "sports", "basketball", "NBA巨星, 篮协主席", "cn"),
    ("易建联", "sports", "basketball", "中国篮球巨星", "cn"),
    ("科比·布莱恩特", "sports", "basketball", "NBA传奇, 5冠王", "us"),
    ("勒布朗·詹姆斯", "sports", "basketball", "NBA球星, 湖人", "us"),
    ("斯蒂芬·库里", "sports", "basketball", "NBA射手, 勇士", "us"),
    ("迈克尔·乔丹", "sports", "basketball", "NBA之神, 公牛王朝", "us"),
    ("凯文·杜兰特", "sports", "basketball", "NBA超级得分手, 太阳", "us"),
    ("字母哥", "sports", "basketball", "NBA巨星, 雄鹿", "gr"),
    ("东契奇", "sports", "basketball", "NBA新星, 独行侠", "si"),
    ("塔图姆", "sports", "basketball", "NBA球星, 凯尔特人", "us"),
    ("周琦", "sports", "basketball", "中国篮球运动员", "cn"),
    ("郭艾伦", "sports", "basketball", "中国篮球后卫", "cn"),

    # ⚽ Sports - Tennis
    ("李娜", "sports", "tennis", "网球大满贯冠军", "cn"),
    ("德约科维奇", "sports", "tennis", "网球巨头", "rs"),
    ("费德勒", "sports", "tennis", "网球传奇", "ch"),
    ("纳达尔", "sports", "tennis", "网球巨头", "es"),
    ("斯维亚泰克", "sports", "tennis", "波兰网球天后", "pl"),
    ("阿尔卡拉斯", "sports", "tennis", "西班牙网球新星", "es"),
    ("彭帅", "sports", "tennis", "中国网球选手", "cn"),

    # ⚽ Sports - F1
    ("舒马赫", "sports", "f1", "F1七冠王", "de"),
    ("汉密尔顿", "sports", "f1", "F1七冠王", "gb"),
    ("维斯塔潘", "sports", "f1", "F1三连冠", "nl"),
    ("勒克莱尔", "sports", "f1", "F1法拉利车手", "mc"),
    ("诺里斯", "sports", "f1", "F1迈凯伦车手", "gb"),
    ("阿隆索", "sports", "f1", "F1传奇车手", "es"),
    ("莱科宁", "sports", "f1", "F1世界冠军", "other"),
    ("维特尔", "sports", "f1", "F1四冠王", "de"),

    # ⚽ Sports - Olympics / Track / Other
    ("刘翔", "sports", "track", "奥运110米栏冠军", "cn"),
    ("郎平", "sports", "volleyball", "女排传奇/教练", "cn"),
    ("谷爱凌", "sports", "skiing", "奥运滑雪冠军", "us"),
    ("苏炳添", "sports", "track", "百米飞人", "cn"),
    ("全红婵", "sports", "diving", "奥运跳水冠军", "cn"),
    ("张雨霏", "sports", "swimming", "奥运游泳冠军", "cn"),
    ("菲尔普斯", "sports", "swimming", "奥运23金游泳传奇", "us"),
    ("博尔特", "sports", "track", "百米世界纪录保持者", "other"),

    # 🥊 Sports - MMA / Boxing
    ("泰森", "sports", "boxing", "拳王", "us"),
    ("梅威瑟", "sports", "boxing", "拳王, 50-0战绩", "us"),
    ("帕奎奥", "sports", "boxing", "拳王, 8级别冠军", "other"),
    ("乌斯曼", "sports", "mma", "UFC次中量级冠军", "other"),
    ("麦格雷戈", "sports", "mma", "UFC超级明星", "other"),

    # 🎮 Sports - eSports
    ("Faker", "sports", "esports", "英雄联盟传奇选手", "kr"),
    ("s1mple", "sports", "esports", "CS:GO顶级选手", "other"),
    ("dev1ce", "sports", "esports", "CS:GO顶级选手", "other"),
    ("ZywOo", "sports", "esports", "CS:GO顶级选手", "fr"),
    ("Ning", "sports", "esports", "英雄联盟世界冠军", "cn"),

    # 🏏 Sports - Cricket
    ("维拉·科利", "sports", "cricket", "印度板球巨星", "in"),
    ("罗希特·夏尔马", "sports", "cricket", "印度板球队长", "in"),
    ("维拉斯·盖伊", "sports", "cricket", "印度板球选手", "in"),

    # 💼 Business
    ("马云", "business", "tech_founder", "阿里巴巴创始人", "cn"),
    ("马化腾", "business", "tech_founder", "腾讯创始人", "cn"),
    ("刘强东", "business", "tech_founder", "京东创始人", "cn"),
    ("张一鸣", "business", "tech_founder", "字节跳动创始人", "cn"),
    ("王健林", "business", "real_estate", "万达集团创始人", "cn"),
    ("王兴", "business", "tech_founder", "美团创始人", "cn"),
    ("雷军", "business", "tech_founder", "小米创始人", "cn"),
    ("董明珠", "business", "industry", "格力电器董事长", "cn"),
    ("马斯克", "business", "tech_ceo", "Tesla/SpaceX CEO", "us"),
    ("贝索斯", "business", "tech_ceo", "亚马逊创始人", "us"),
    ("比尔·盖茨", "business", "tech_ceo", "微软创始人", "us"),
    ("蒂姆·库克", "business", "tech_ceo", "苹果CEO", "us"),
    ("扎克伯格", "business", "tech_ceo", "Meta CEO", "us"),
    ("巴菲特", "business", "investor", "伯克希尔CEO", "us"),
    ("索罗斯", "business", "investor", "量子基金创始人", "us"),
    ("许家印", "business", "real_estate", "恒大集团创始人", "cn"),

    # 🎵 Music - Pop/Rock US/UK/Legend
    ("Michael Jackson", "music", "music/legend", "King of Pop, Thriller", "us"),
    ("Elvis Presley", "music", "music/legend", "The King of Rock and Roll", "us"),
    ("Madonna", "music", "music/pop", "Queen of Pop", "us"),
    ("Beyonce", "music", "music/pop", "Global pop icon", "us"),
    ("Taylor Swift", "music", "music/pop", "Pop/country superstar", "us"),
    ("Ariana Grande", "music", "music/pop", "Pop vocalist", "us"),
    ("Drake", "music", "music/hiphop", "Hip-hop/pop star", "us"),
    ("Billie Eilish", "music", "music/pop", "Alt-pop star", "us"),
    ("Lady Gaga", "music", "music/pop", "Pop/electronic star", "us"),
    ("Ed Sheeran", "music", "music/pop", "British pop singer-songwriter", "uk"),
    ("Adele", "music", "music/pop", "British soul/pop singer", "uk"),
    ("Coldplay", "music", "music/band", "British rock band", "uk"),
    ("Queen", "music", "music/band", "British rock legends", "uk"),
    ("The Beatles", "music", "music/band", "British rock legends", "uk"),
    ("One Direction", "music", "music/band", "British boy band", "uk"),
    ("Little Mix", "music", "music/band", "British girl group", "uk"),
    ("Sam Smith", "music", "music/pop", "British pop/soul singer", "uk"),
    ("Dua Lipa", "music", "music/pop", "British pop star", "uk"),

    # 🎵 Music - Electronic/DJ
    ("Calvin Harris", "music", "music/electronic", "Scottish DJ/producer", "uk"),
    ("David Guetta", "music", "music/electronic", "French DJ/producer", "fr"),
    ("Avicii", "music", "music/electronic", "Swedish DJ, 2018 deceased", "se"),
    ("Martin Garrix", "music", "music/electronic", "Dutch DJ/producer", "nl"),
    ("Tiësto", "music", "music/electronic", "Dutch DJ legend", "nl"),
    ("Marshmello", "music", "music/electronic", "American DJ/producer", "us"),
    ("Kygo", "music", "music/electronic", "Norwegian DJ/producer", "no"),
    ("Deadmau5", "music", "music/electronic", "Canadian DJ/producer", "ca"),

    # 🎵 Music - Classical/Jazz
    ("郎朗", "music", "music/classical", "Pianist, world-renowned", "cn"),
    ("李云迪", "music", "music/classical", "Pianist, Chopin competition winner", "cn"),
    ("马克西姆", "music", "music/classical", "Pianist,克罗地亚/中国", "hr"),
    ("帕尔曼", "music", "music/classical", "Violinist, Israeli virtuoso", "il"),
    ("马友友", "music", "music/classical", "Cellist, world-renowned", "us"),
    ("Louis Armstrong", "music", "music/classical", "Jazz legend, 1971 deceased", "us"),
    ("Miles Davis", "music", "music/classical", "Jazz legend, 1991 deceased", "us"),
    ("Ella Fitzgerald", "music", "music/classical", "Jazz legend, 1996 deceased", "us"),

    # 🎵 Music - Hip-Hop/Rap US
    ("Jay-Z", "music", "music/hiphop", "Hip-hop mogul", "us"),
    ("Kanye West", "music", "music/hiphop", "Hip-hop/producer", "us"),
    ("Eminem", "music", "music/hiphop", "Rap legend", "us"),
    ("Kendrick Lamar", "music", "music/hiphop", "Pulitzer-winning rapper", "us"),
    ("Travis Scott", "music", "music/hiphop", "Hip-hop artist", "us"),
    ("Post Malone", "music", "music/hiphop", "Hip-hop/pop artist", "us"),

    # 🎵 Music - Chinese/CN/TW/HK Pop
    ("周杰伦", "music", "music/pop", "华语流行天王", "tw"),
    ("王力宏", "music", "music/pop", "华语流行歌手", "tw"),
    ("林俊杰", "music", "music/pop", "新加坡/华语流行歌手", "sg"),
    ("张学友", "music", "music/pop", "香港歌神", "hk"),
    ("张靓颖", "music", "music/pop", "华语流行歌手", "cn"),
    ("邓紫棋", "music", "music/pop", "香港唱作歌手", "hk"),
    ("五月天", "music", "music/band", "台湾摇滚乐团", "tw"),
    ("苏打绿", "music", "music/band", "台湾乐团", "tw"),
    ("TFBOYS", "music", "music/band", "中国少年组合", "cn"),
    ("王菲", "music", "music/pop", "华语天后", "cn"),
    ("崔健", "music", "music/rock", "中国摇滚之父", "cn"),
    ("李宗盛", "music", "music/pop", "台湾音乐教父", "tw"),
    ("罗大佑", "music", "music/pop", "台湾音乐之父", "tw"),

    # 🎵 Music - Chinese Rap
    ("MC Hotdog", "music", "music/rap", "台湾说唱歌手", "tw"),
    ("潘玮柏", "music", "music/rap", "华语说唱/流行", "tw"),
    ("GAI", "music", "music/rap", "中国说唱歌手", "cn"),

    # 🎵 Music - Korean Pop
    ("BTS", "music", "music/band", "Korean boy band, global superstars", "kr"),
    ("BLACKPINK", "music", "music/band", "Korean girl group", "kr"),
    ("EXO", "music", "music/band", "Korean boy band", "kr"),
    ("PSY", "music", "music/pop", "Korean singer, Gangnam Style", "kr"),
    ("BigBang", "music", "music/band", "Korean boy band", "kr"),
    ("IU", "music", "music/pop", "Korean solo singer star", "kr"),
    ("权志龙", "music", "music/hiphop", "韩国说唱/BigBang", "kr"),

    # 🎵 Music - Latin/World Pop
    ("Shakira", "music", "music/pop", "Colombian pop star", "co"),
    ("Ricky Martin", "music", "music/pop", "Puerto Rican pop star", "pr"),
    ("Selena Gomez", "music", "music/pop", "American/Mexican pop star", "us"),
    ("Justin Bieber", "music", "music/pop", "Canadian pop star", "ca"),
    ("Shawn Mendes", "music", "music/pop", "Canadian pop singer", "ca"),
    ("Bruno Mars", "music", "music/pop", "American pop/R&B star", "us"),
    ("Camila Cabello", "music", "music/pop", "Cuban-American pop star", "us"),
    ("Bad Bunny", "music", "music/pop", "Puerto Rican reggaeton star", "pr"),
    ("J Balvin", "music", "music/pop", "Colombian reggaeton star", "co"),

    # 🔥 Hot News - Influencers/Social Media Stars (2024-2025)
    ("MrBeast", "hotnews", "hotnews/influencer", "YouTube mega-influencer, MrBeast Burger / Beast Games", "us"),
    ("Logan Paul", "hotnews", "hotnews/influencer", "YouTube star, boxer, Prankster", "us"),
    ("KSI", "hotnews", "hotnews/influencer", "UK YouTube star, boxer, music artist", "uk"),
    ("Charli D'Amelio", "hotnews", "hotnews/influencer", "TikTok queen, 150M+ followers", "us"),
    ("Addison Rae", "hotnews", "hotnews/influencer", "TikTok star, actress", "us"),
    ("Bella Poarch", "hotnews", "hotnews/influencer", "TikTok star, singer", "us"),
    ("Markiplier", "hotnews", "hotnews/influencer", "YouTube gaming influencer", "us"),
    ("SSSniperWolf", "hotnews", "hotnews/influencer", "YouTube content creator", "us"),
    ("李子柒", "hotnews", "hotnews/influencer", "Chinese rural lifestyle influencer, YouTube star", "cn"),
    ("疯狂小杨哥", "hotnews", "hotnews/influencer", "Chinese live-streaming star, comedy带货", "cn"),
    ("李佳琦", "hotnews", "hotnews/influencer", "Chinese top beauty streamer", "cn"),
    ("辛巴", "hotnews", "hotnews/influencer", "Chinese live-streaming king, SIMBA", "cn"),
    ("董宇辉", "hotnews", "hotnews/influencer", "Chinese edu-streamer, Oriental Selection", "cn"),
    ("薇娅", "hotnews", "hotnews/influencer", "Chinese top live-streamer, 2021 tax scandal", "cn"),

    # 🔥 Hot News - 2024-2025 Breakout Stars
    ("贾玲", "hotnews", "hotnews/breakout_star", "Director/Actress, YOLO 热辣滚烫", "cn"),
    ("饺子", "hotnews", "hotnews/breakout_star", "Director of 哪吒之魔童闹海 NEZHA 2", "cn"),

    # 🔥 Hot News - Tech/Business Leaders
    ("Sam Altman", "hotnews", "hotnews/tech", "OpenAI CEO, ChatGPT creator", "us"),
    ("Satya Nadella", "hotnews", "hotnews/tech", "Microsoft CEO", "us"),
    ("Jensen Huang", "hotnews", "hotnews/tech", "NVIDIA CEO, AI chip king", "us"),

    # 🔥 Hot News - AI/Crypto
    ("CZ", "hotnews", "hotnews/ai_creator", "Binance founder, crypto mogul", "other"),
    ("SBF", "hotnews", "hotnews/viral_moment", "FTX founder, crypto scandal figure", "us"),

    # 📺 TV - Chinese Mainland Hosts
    ("何炅", "tv", "tv/host", "快乐大本营主持人", "cn"),
    ("汪涵", "tv", "tv/host", "天天向上主持人", "cn"),
    ("康辉", "tv", "tv/news_anchor", "新闻联播主播", "cn"),
    ("董卿", "tv", "tv/host", "诗词大会/春晚主持人", "cn"),
    ("撒贝宁", "tv", "tv/host", "今日说法/明星大侦探主持人", "cn"),
    ("谢娜", "tv", "tv/host", "快乐大本营主持人", "cn"),
    ("李维嘉", "tv", "tv/host", "快乐大本营主持人", "cn"),
    ("吴昕", "tv", "tv/host", "快乐大本营主持人", "cn"),
    ("杜海涛", "tv", "tv/host", "快乐大本营主持人", "cn"),
    ("陈鲁豫", "tv", "tv/host", "鲁豫有约主持人", "cn"),
    ("许知远", "tv", "tv/host", "十三邀主持人", "cn"),
    ("李咏", "tv", "tv/legend", "非常6+1/春晚主持人", "cn"),
    ("方琼", "tv", "tv/host", "快乐星球主持人", "cn"),
    ("金炜", "tv", "tv/host", "IP频道主持人", "cn"),
    ("程雷", "tv", "tv/host", "达人秀主持人", "cn"),
    ("陈蓉", "tv", "tv/host", "相约星期六主持人", "cn"),
    ("骆新", "tv", "tv/host", "梦想改造家主持人", "cn"),
    ("刘维", "tv", "tv/variety", "百变大咖秀明星", "cn"),
    ("董倩", "tv", "tv/news_anchor", "央视新闻主播", "cn"),
    ("王宁", "tv", "tv/news_anchor", "央视主播", "cn"),
    ("水均益", "tv", "tv/news_anchor", "央视主播", "cn"),
    ("白岩松", "tv", "tv/news_anchor", "央视主播", "cn"),
    ("敬一丹", "tv", "tv/news_anchor", "央视主播", "cn"),

    # 📺 TV - Taiwan Hosts
    ("蔡康永", "tv", "tv/host", "康熙来了主持人", "tw"),
    ("小S 徐熙娣", "tv", "tv/host", "康熙来了主持人", "tw"),
    ("沈玉琳", "tv", "tv/variety", "综艺大哥大明星", "tw"),
    ("陶晶莹", "tv", "tv/host", "台湾主持人", "tw"),
    ("大S 徐熙媛", "tv", "tv/actor", "台湾演员/流星花园", "tw"),

    # 📺 TV - Korean Hosts/Variety
    ("刘在锡", "tv", "tv/host", "Running Man主持人", "kr"),
    ("姜虎东", "tv", "tv/variety", "新西游记MC", "kr"),
    ("李光洙", "tv", "tv/variety", "Running Man固定嘉宾", "kr"),
    ("HAHA", "tv", "tv/variety", "Running Man固定嘉宾", "kr"),
    ("金钟国", "tv", "tv/variety", "Running Man固定嘉宾", "kr"),
    ("池石镇", "tv", "tv/variety", "Running Man固定嘉宾", "kr"),
    ("宋智孝", "tv", "tv/variety", "Running Man固定嘉宾", "kr"),
    ("李多熙", "tv", "tv/host", "韩国TV主持人", "kr"),
    ("申东烨", "tv", "tv/host", "我的大叔等MC", "kr"),
    ("朴娜莱", "tv", "tv/variety", "Mast节目明星", "kr"),

    # 📺 TV - Japanese Hosts
    ("香取慎吾", "tv", "tv/variety", "岚成员/综艺偶像", "jp"),
    ("草彅刚", "tv", "tv/variety", "岚成员/综艺偶像", "jp"),
    ("稲垣吾郎", "tv", "tv/variety", "岚成员/综艺偶像", "jp"),
    ("北大路欣也", "tv", "tv/host", "日本主持人", "jp"),

    # 👗 Fashion / Models
    ("刘雯", "fashion", "fashion/model", "Chinese supermodel, 国际超模", "cn"),
    ("奚梦瑶", "fashion", "fashion/model", "Chinese model, 维密天使", "cn"),
    ("秦舒培", "fashion", "fashion/model", "Chinese model", "cn"),
    ("雎晓雯", "fashion", "fashion/model", "Chinese model, 维密天使", "cn"),
    ("贺聪", "fashion", "fashion/model", "Chinese model", "cn"),
    ("Kendall Jenner", "fashion", "fashion/model", "American supermodel, 卡戴珊家族", "us"),
    ("Gigi Hadid", "fashion", "fashion/model", "American model", "us"),
    ("Bella Hadid", "fashion", "fashion/model", "American model", "us"),
    ("Naomi Campbell", "fashion", "fashion/model", "British supermodel, 黑珍珠", "uk"),
    ("Cindy Crawford", "fashion", "fashion/model", "American supermodel, 传奇超模", "us"),
    ("Gisele Bündchen", "fashion", "fashion/model", "Brazilian supermodel", "br"),
    ("水原希子", "fashion", "fashion/model", "Japanese-American model/actress", "jp"),
    ("Iriana", "fashion", "fashion/model", "Korean model/Iriana LJ", "kr"),
    ("胡兵", "fashion", "fashion/model", "Chinese actor/model", "cn"),
    ("吴亦凡", "fashion", "fashion/model", "Chinese actor/rapper/model", "cn"),
    ("赵磊", "fashion", "fashion/model", "Chinese male model", "cn"),
    ("金大川", "fashion", "fashion/model", "Chinese male model", "cn"),
    ("裴斗娜", "fashion", "fashion/model", "Korean actress/model", "kr"),
    ("韩聪聪", "fashion", "fashion/model", "Chinese model", "cn"),
    ("Liu Yifei", "fashion", "fashion/model", "Chinese actress/model, 刘亦菲", "cn"),

    # 🎮 Gaming / Streamers
    ("Pokimane", "gaming", "gaming/streamer", "Twitch streamer, Fortnite/Valorant", "us"),
    ("Ninja", "gaming", "gaming/streamer", "Twitch/Fortnite legend", "us"),
    ("Shroud", "gaming", "gaming/streamer", "Twitch streamer, FPS games", "ca"),
    ("DrLupo", "gaming", "gaming/streamer", "Twitch variety streamer", "us"),
    ("TimtheTatman", "gaming", "gaming/streamer", "Twitch variety streamer", "us"),
    ("xQc", "gaming", "gaming/streamer", "Canadian Twitch star, Overwatch/Valorant", "ca"),
    ("Valkyrae", "gaming", "gaming/streamer", "YouTube/Gaming star, Valorant", "us"),
    ("Markiplier", "gaming", "gaming/youtuber", "YouTube gaming influencer", "us"),
    ("PewDiePie", "gaming", "gaming/youtuber", "Swedish YouTube legend, gaming/comedy", "se"),
    ("SSSniperWolf", "gaming", "gaming/youtuber", "YouTube gaming star", "us"),
    ("Kai Cenat", "gaming", "gaming/streamer", "Twitch king, variety/roleplay", "us"),
    ("Speed", "gaming", "gaming/streamer", "Twitch GTA/roleplay star", "us"),
    ("Dr Disrespect", "gaming", "gaming/streamer", "Twitch gaming streamer, Bearded Bod", "us"),
    ("Myth", "gaming", "gaming/streamer", "Twitch Fortnite/Apex streamer", "us"),
    ("Lirik", "gaming", "gaming/streamer", "Twitch variety streamer", "us"),
    (" Summit1g", "gaming", "gaming/streamer", "Twitch CS:GO streamer", "us"),
    (" dakidlock", "gaming", "gaming/streamer", "Twitch streamer", "us"),
    ("Mongraal", "gaming", "gaming/streamer", "Fortnite pro/streamer", "uk"),
    ("Tfue", "gaming", "gaming/streamer", "Fortnite legend, Twitch star", "us"),
    ("NICKMERCS", "gaming", "gaming/streamer", "Twitch/FPS gaming star", "us"),

    # 🍳 Food / Chefs
    ("Gordon Ramsay", "food", "food/chef", "British celebrity chef, Hell's Kitchen", "uk"),
    ("Jamie Oliver", "food", "food/chef", "British celebrity chef, food campaigner", "uk"),
    ("詹姆斯·布里克曼", "food", "food/chef", "British celebrity chef, MasterChef", "uk"),
    ("Rachael Ray", "food", "food/chef", "American celebrity chef, TV host", "us"),
    ("Martha Stewart", "food", "food/chef", "American lifestyle/culinary queen", "us"),
    ("Anthony Bourdain", "food", "food/chef", "American chef/travel host, 2018 deceased", "us"),
    ("Rick Bayless", "food", "food/chef", "American chef, Mexican cuisine", "us"),
    ("Giada De Laurentiis", "food", "food/chef", "American chef, Italian cuisine", "us"),
    ("Wolfgang Puck", "food", "food/chef", "Austrian-American celebrity chef", "us"),
    ("Emeril Lagasse", "food", "food/chef", "American celebrity chef, TV personality", "us"),
    ("Ina Garten", "food", "food/chef", "American cookbook author, Barefoot Contessa", "us"),
    ("王刚", "food", "food/chef", "Chinese food blogger/chef, 老饭骨", "cn"),
    ("美食作家王刚", "food", "food/chef", "Chinese chef/YouTuber", "cn"),
    ("绵羊料理", "food", "food/youtuber", "Japanese food YouTuber", "jp"),
    ("Binging with Babish", "food", "food/youtuber", "American food/YouTube star", "us"),
    ("Kenji Lopez-Alt", "food", "food/youtuber", "American food scientist/YouTuber", "us"),

    # 🎨 Art / Contemporary Artists
    ("村上隆", "art", "art/contemporary", "Japanese contemporary artist, Superflat", "jp"),
    ("草间弥生", "art", "art/contemporary", "Japanese artist, 圆点女王", "jp"),
    ("曾梵志", "art", "art/painter", "Chinese contemporary painter", "cn"),
    ("张晓刚", "art", "art/painter", "Chinese contemporary painter", "cn"),
    ("杰夫·昆斯", "art", "art/contemporary", "American contemporary artist", "us"),
    ("班克斯", "art", "art/contemporary", "British street artist, Banksy", "uk"),
    ("达明安·赫斯特", "art", "art/contemporary", "British contemporary artist", "uk"),
    ("蜷川实花", "art", "art/photographer", "Japanese photographer/director", "jp"),
    ("KAWS", "art", "art/contemporary", "American artist, KAWS figures", "us"),
    ("奈良美智", "art", "art/contemporary", "Japanese artist, Angry小女孩", "jp"),
    ("Jean-Michel Basquiat", "art", "art/contemporary", "American artist, 1988 deceased", "us"),
    ("杰夫·昆斯", "art", "art/contemporary", "American contemporary artist, 气球狗", "us"),

    # 🔬 Academic / Science
    ("爱因斯坦", "academic", "academic/scientist", "Albert Einstein, 物理学家, 1955 deceased", "de"),
    ("霍金", "academic", "academic/scientist", "Stephen Hawking, 物理学家, 2018 deceased", "uk"),
    ("杨振宁", "academic", "academic/scientist", "Chinese physicist, Nobel laureate", "cn"),
    ("屠呦呦", "academic", "academic/scientist", "Chinese scientist, Nobel Medicine 2015", "cn"),
    ("袁隆平", "academic", "academic/scientist", "Chinese agronomist, 杂交水稻之父, 2021 deceased", "cn"),
    ("钟南山", "academic", "academic/medical", "Chinese respiratory disease expert", "cn"),
    ("张文宏", "academic", "academic/medical", "Chinese epidemiologist, COVID-19 expert", "cn"),
    ("Neil deGrasse Tyson", "academic", "academic/scientist", "American astrophysicist, Cosmos host", "us"),
    ("Carl Sagan", "academic", "academic/scientist", "American astronomer, Cosmos, 1996 deceased", "us"),
    ("Jane Goodall", "academic", "academic/scientist", "British primatologist", "uk"),
    ("颜宁", "academic", "academic/scientist", "Chinese structural biologist", "cn"),
    ("施一公", "academic", "academic/scientist", "Chinese structural biologist", "cn"),
    ("韩礼德", "academic", "academic/scientist", "Chinese-American linguist", "us"),
    ("李开复", "academic", "academic/tech_thinker", "Chinese AI expert, author", "cn"),

    # 🎞️ Anime / Manga / Voice Actors
    ("宫崎骏", "anime", "anime/director", "Japanese anime film director, 奥斯卡获奖", "jp"),
    ("尾田荣一郎", "anime", "anime/manga_artist", "Manga artist, One Piece", "jp"),
    ("岸本齐史", "anime", "anime/manga_artist", "Manga artist, Naruto", "jp"),
    ("鸟山明", "anime", "anime/manga_artist", "Manga artist, Dragon Ball", "jp"),
    ("新海诚", "anime", "anime/director", "Anime film director, Your Name", "jp"),
    ("青山刚昌", "anime", "anime/manga_artist", "Manga artist, Detective Conan", "jp"),
    ("荒川弘", "anime", "anime/manga_artist", "Manga artist, Fullmetal Alchemist", "jp"),
    ("富坚义博", "anime", "anime/manga_artist", "Manga artist, Hunter x Hunter", "jp"),
    ("ONE", "anime", "anime/manga_artist", "Web manga artist, One-Punch Man", "jp"),
    ("刘小光", "anime", "anime/voice_actor", "Chinese voice actor", "cn"),
    ("山寺宏一", "anime", "anime/voice_actor", "Japanese voice actor", "jp"),
    ("田中真弓", "anime", "anime/voice_actor", "Japanese voice actress", "jp"),
    ("绪方惠美", "anime", "anime/voice_actor", "Japanese voice actress", "jp"),
    ("林原惠", "anime", "anime/voice_actor", "Japanese voice actress", "jp"),
    ("斯科特·艾化", "anime", "anime/voice_actor", "American voice actor", "us"),

    # 📺 TV - US Hosts
    ("Jimmy Fallon", "tv", "tv/host", "Tonight Show主持人", "us"),
    ("Stephen Colbert", "tv", "tv/host", "Colbert Report主持人", "us"),
    ("Trevor Noah", "tv", "tv/host", "Daily Show主持人", "us"),
    ("Ellen DeGeneres", "tv", "tv/host", "Ellen Show主持人", "us"),
    ("Drew Carey", "tv", "tv/host", "Price is Right主持人", "us"),
    ("Pat Sajak", "tv", "tv/host", "Wheel of Fortune主持人", "us"),
    ("Alex Trebek", "tv", "tv/legend", "Jeopardy传奇主播", "us"),

    # 📺 TV - Reality Stars
    ("蔡徐坤", "tv", "tv/reality_star", "偶像练习生冠军", "cn"),
    ("王鹤棣", "tv", "tv/actor", "流星花园主演", "cn"),
    ("吴宣仪", "tv", "tv/reality_star", "创造101选手", "cn"),
    ("孟美岐", "tv", "tv/reality_star", "创造101冠军", "cn"),
    ("杨超越", "tv", "tv/reality_star", "创造101选手/火箭少女", "cn"),
    ("Lisa", "tv", "tv/reality_star", "BLACKPINK/Youth With You导师", "kr"),
    ("宁静", "tv", "tv/reality_star", "乘风破浪的姐姐", "cn"),
    ("伊能静", "tv", "tv/reality_star", "乘风破浪的姐姐", "cn"),
    ("黄圣依", "tv", "tv/reality_star", "乘风破浪的姐姐", "cn"),

    # 📺 TV - US Media/Podcast Hosts (NEW)
    ("Joe Rogan", "tv", "tv/podcast", "Podcast host, comedian, JRE", "us"),
    ("Tucker Carlson", "tv", "tv/media", "Fox News host, political commentator", "us"),

    # 💼 Business - Tech CEOs (NEW)
    ("Tim Cook", "business", "tech_ceo", "Apple CEO", "us"),

    # 🔥 Hot News - Influencers (NEW)
    ("Kim Kardashian", "hotnews", "hotnews/influencer", "Reality TV star, Skims founder", "us"),

    # 🔥 Hot News - Fashion (NEW)
    ("Giles Deacon", "hotnews", "hotnews/fashion", "British fashion designer", "uk"),
    ("Valentino Garavani", "hotnews", "hotnews/fashion", "Italian fashion designer", "it"),

    # 🎬 Film - Bollywood (NEW - Shah Rukh Khan already present, adding Amir Khan)
    ("阿米尔·汗", "film", "film/legend", "Indian actor, Bollywood star", "in"),
    ("沙鲁克·汗", "film", "film/legend", "Indian actor, Bollywood King", "in"),

    # 🎵 Music - Global Pop (NEW)
    ("蕾哈娜", "music", "music/pop", "Global pop star, Barbados", "bb"),

    # 🌍 Politics - Additional global figures (NEW - some overlap with POLITICS list)
    ("Donald Trump Jr.", "politics", "politics/current", "US political figure, Trump son", "us"),
    ("梅洛尼", "politics", "politics/current", "意大利总理", "it"),
    ("博索纳罗", "politics", "politics/former", "巴西前总统", "br"),
    ("拉马福萨", "politics", "politics/current", "南非总统", "za"),
]

def get_db():
    """Get database connection."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def celebrity_hash(name, category):
    """Generate unique hash for a celebrity."""
    return hashlib.sha256(f"{name}{category}{time.time()}".encode()).hexdigest()[:16]

def check_celebrity_exists(cursor, name, category):
    """Check if celebrity already exists in database."""
    cursor.execute(
        "SELECT id FROM faces WHERE name=? AND is_celebrity=1 AND status='active' LIMIT 1",
        (name,)
    )
    row = cursor.fetchone()
    return row['id'] if row else None

def init_celebrity_schema():
    """Initialize database schema for celebrity data."""
    conn = get_db()
    c = conn.cursor()

    # Create celebrity_info table for additional metadata
    c.execute("""
        CREATE TABLE IF NOT EXISTS celebrity_info (
            id INTEGER PRIMARY KEY,
            face_id INTEGER REFERENCES faces(id),
            category TEXT NOT NULL,
            subcategory TEXT,
            country TEXT,
            description TEXT,
            wikipedia_url TEXT,
            image_source TEXT,
            embedding_computed INTEGER DEFAULT 0,
            embedding_model TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(face_id)
        )
    """)

    # Add risk_level column if it doesn't exist (for political figures)
    try:
        c.execute("ALTER TABLE celebrity_info ADD COLUMN risk_level TEXT DEFAULT 'warning'")
        logger.info("Added risk_level column to celebrity_info")
    except sqlite3.OperationalError:
        pass  # Column already exists

    conn.commit()
    conn.close()
    logger.info("Celebrity schema initialized")

def add_celebrity(name, category, subcategory, description, country, risk_level='warning'):
    """Add a celebrity to the database (face + metadata)."""
    conn = get_db()
    c = conn.cursor()

    # Check if already exists
    existing_id = check_celebrity_exists(c, name, category)
    if existing_id:
        conn.close()
        return existing_id, "already_exists"

    # Hash ID
    hid = celebrity_hash(name, category)

    # Insert into faces table
    c.execute("""
        INSERT INTO faces (name, description, hash_id, is_celebrity, uploader_id,
                          original_price, ai_declaration, status, category)
        VALUES (?, ?, ?, 1, NULL, 0.0, 1, 'active', ?)
    """, (name, description, hid, category))

    face_id = c.lastrowid

    # Insert into celebrity_info
    c.execute("""
        INSERT OR IGNORE INTO celebrity_info
        (face_id, category, subcategory, country, description, risk_level)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (face_id, category, subcategory, country, description, risk_level))

    conn.commit()
    conn.close()

    logger.info(f"Added celebrity: {name} (face_id={face_id}, risk={risk_level})")
    return face_id, "added"


def add_political(name, subcategory, description, country, risk_level='warning'):
    """Add a political figure for risk detection."""
    return add_celebrity(name, 'politics', subcategory, description, country, risk_level)


def build_politics_database():
    """Build the politics database for risk detection."""
    logger.info(f"Building politics database with {len(POLITICS)} entries...")

    init_celebrity_schema()

    results = {"added": 0, "exists": 0, "errors": 0, "blocked": 0, "warning": 0}

    for name, category, subcategory, description, country, risk_level in POLITICS:
        try:
            face_id, status = add_celebrity(name, category, subcategory, description, country, risk_level)
            if status == "added":
                results["added"] += 1
                if risk_level == 'blocked':
                    results["blocked"] += 1
                else:
                    results["warning"] += 1
            else:
                results["exists"] += 1
        except Exception as e:
            logger.error(f"Error adding {name}: {e}")
            results["errors"] += 1

    logger.info(f"Politics database build complete: {results}")
    return results

def build_celebrity_database():
    """Build the celebrity database from the CELEBRITIES list."""
    logger.info(f"Building celebrity database with {len(CELEBRITIES)} entries...")

    init_celebrity_schema()

    results = {"added": 0, "exists": 0, "errors": 0}

    for name, category, subcategory, description, country in CELEBRITIES:
        try:
            face_id, status = add_celebrity(name, category, subcategory, description, country)
            if status == "added":
                results["added"] += 1
            else:
                results["exists"] += 1
        except Exception as e:
            logger.error(f"Error adding {name}: {e}")
            results["errors"] += 1

    logger.info(f"Database build complete: {results}")
    return results

def list_celebrities_without_embeddings():
    """List all celebrities that don't have embeddings yet."""
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        SELECT f.id, f.name, ci.category, ci.subcategory, ci.description, ci.country
        FROM faces f
        JOIN celebrity_info ci ON f.id = ci.face_id
        LEFT JOIN face_embeddings fe ON f.id = fe.face_id
        WHERE fe.id IS NULL AND f.status='active' AND f.is_celebrity=1
    """)
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def list_all_celebrities():
    """List all celebrities with embedding status."""
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        SELECT f.id, f.name, f.description, ci.category, ci.subcategory, ci.country,
               fe.id as has_embedding, fe.model_name
        FROM faces f
        JOIN celebrity_info ci ON f.id = ci.face_id
        LEFT JOIN face_embeddings fe ON f.id = fe.face_id
        WHERE f.status='active' AND f.is_celebrity=1
        ORDER BY ci.category, f.name
    """)
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "build":
            build_celebrity_database()
        elif cmd == "build_politics":
            r = build_politics_database()
            print(f"Politics: added={r['added']}, exists={r['exists']}, blocked={r['blocked']}, warning={r['warning']}, errors={r['errors']}")
        elif cmd == "pending":
            pending = list_celebrities_without_embeddings()
            print(f"Celebrities without embeddings: {len(pending)}")
            for p in pending[:10]:
                print(f"  - {p['name']} ({p['category']})")
        elif cmd == "list":
            all_cels = list_all_celebrities()
            print(f"Total celebrities: {len(all_cels)}")
            for c in all_cels[:20]:
                emb_status = "✓" if c['has_embedding'] else "✗"
                print(f"  [{emb_status}] {c['name']} - {c['category']}")
        else:
            print(f"Unknown command: {cmd}")
            print("Usage: python celebrity_db_builder.py [build|build_politics|pending|list]")
    else:
        build_celebrity_database()