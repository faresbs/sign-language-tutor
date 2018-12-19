#
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import webbrowser
import random

from kivy.app import App
from kivy.clock import Clock
from kivy.uix.popup import Popup
from kivy.uix.button import Button
from kivy.core.window import Window
from kivy.core.text import LabelBase
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.screenmanager import Screen
from kivy.properties import ObjectProperty
from kivy.utils import get_color_from_hex

#from kivy.uix.camera import Camera 

from arithmetic import Arithmetic, json_settings

#-----pour la webam---
#from kivy.lang import Builder

from kivy.uix.image import Image
from kivy.graphics.texture import Texture
import cv2


#Recognition predictor
import model as rec

#from camCapture import camCapture

# Color the background
Window.clearcolor = get_color_from_hex("#300000")

# Register fonts
#LabelBase.register(
#    name="Roboto",
#    fn_regular="./fonts/Roboto-Thin.ttf",
#    fn_bold="./fonts/Roboto-Medium.ttf"
#)

################################################################################
class KivyCamera(Image):
    def __init__(self, capture, fps, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = capture
        Clock.schedule_interval(self.update, 1.0 / fps)
    
    def update(self, dt):
        
        while(True):
            ret, frame = self.capture.read()
            cv2.imshow('frame',frame)

            print('here')

            #Call predictor
            print (rec.predict(frame))

            with open('hey.txt','a') as f: f.write(rec.predict(frame)+"\n")

            if cv2.waitKey(1) & 0xFF == ord('q'):#to get out from the infinite loop
                break


class CamApp(App):
    def build(self):
        self.capture = cv2.VideoCapture(0)

        self.my_camera = KivyCamera(capture=self.capture, fps=30)
        return self.my_camera

    def on_stop(self):
        #without this, app will not exit even if the window is closed
        self.capture.release()

class KivyTutorRoot(BoxLayout):
    """
    Root of all widgets
    """
    hmi_screen = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(KivyTutorRoot, self).__init__(**kwargs)
        # List of previous screens
        self.screen_list = []
        self.is_mix = False
        self.hmi_popup = HmiPopup()
        
    def changeScreen(self, next_screen):
       
        operations = "addition Novice Average Experienced".split()
        images = "1.png".split()
        question = None

        # If screen is not already in the list fo prevous screens
        if self.ids.kivy_screen_manager.current not in self.screen_list:
            self.screen_list.append(self.ids.kivy_screen_manager.current)

        if next_screen == "about this app":
            self.ids.kivy_screen_manager.current = "about_screen"

        else:
            self.hmi_screen.question_text.text = "Start with this sign dude!!!"
            
            idx = random.randint(0, len(images) - 1)     
            #image = images[idx]
            #idx += 1  
            image = images[idx]
            #inc = increment()
            self.hmi_screen.image.source = image
            #cam = Camera()
            cam = CamApp()
            #camCapture()
            self.ids.kivy_screen_manager.current = "hmi_screen"
    
    def changeScreen_(self):
        images = "1.png 2.png 3.jpeg 4.png 5.jpg 6.jpg".split()
        idx = self.hmi_screen.button.idx 
        if idx < len(images):
            image = images[idx]
            self.hmi_screen.image.source = image            
            self.hmi_screen.button.idx += 1  
        else:
            self.hmi_screen.button.idx = 0
            idx = self.hmi_screen.button.idx
            image = images[idx]
            self.hmi_screen.image.source = image
            self.hmi_screen.button.idx = 1
            
    def onBackBtn(self):
        # Check if there are any screen to go back to
        if self.screen_list:
            # if there are screens we can go back to, the just do it
            self.ids.kivy_screen_manager.current = self.screen_list.pop()
            # Saw we don't want to close
            return True
        # No more screens to go back to
        return False


################################################################################
class HmiScreen(Screen, Arithmetic):
    
    #Widget that will arc as a screen and hold funcs for hmi questions
    def __init__(self, *args, **kwargs):
        super(HmiScreen, self).__init__(*args, **kwargs)

################################################################################
class HmiPopup(Popup):
    
    #Popup for telling user whether he got it right or wrong
    GOOD = "{} :D"
    BAD = "{} Try again!!"
    GOOD_LIST = "Awesome! Amazing! Excellent! Correct!".split()
    BAD_LIST = ["Almost!", "Close!", "Sorry!", "False!"]

    message = ObjectProperty()
    wrapped_button = ObjectProperty()

    def __init__(self, *args, **kwargs):
        super(HmiPopup, self).__init__(*args, **kwargs)
    
    def open(self, correct=True):
        # If answer is correct take off button if its visible
        if correct:
            if self.wrapped_button in self.content.children:
                self.content.remove_widget(self.wrapped_button)
        # If answers is wrong, display button if not visible
        else:
            if self.wrapped_button not in self.content.children:
                self.content.add_widget(self.wrapped_button)

        # Set up text message
        self.message.text = self._prep_text(correct)

        # display popup
        super(HmiPopup, self).open()
        if correct:
            Clock.schedule_once(self.dismiss, 1)

    def _prep_text(self, correct):
        if correct:
            index = random.randint(0, len(self.GOOD_LIST) - 1)
            return self.GOOD.format(self.GOOD_LIST[index])
        else:
            index = random.randint(0, len(self.BAD_LIST) - 1)
            hmi_screen = App.get_running_app().root.hmi_screen
            return self.BAD.format(self.BAD_LIST[index])

################################################################################
class KeyPad(GridLayout):
    
    #Documentation for KeyPad
    def __init__(self, *args, **kwargs):
        super(KeyPad, self).__init__(*args, **kwargs)
        self.cols = 3
        self.spacing = 10
        self.createButtons()

    def createButtons(self):
        _list = ["Yes", "No", "GO!"]
        for num in _list:
            self.add_widget(Button(text=str(num), on_release=self.onBtnPress))
                       
    def onBtnPress(self, btn):
        hmi_screen = App.get_running_app().root.ids.hmi_screen
        #####################################
        #####################################
        ####################################
        #!dans ce cas, le système doit retourner un yes ou no en se basant sur cela on saura si si true or false, donc l'application lira plus la réponse du clavier mais du système, donc ça devient
        #answer_text = système.answer
        #######################################
        ########################################
        #########################################
        answer_text = hmi_screen.answer_text

        
        #answer_text.text in this case will have yes or no as value 
        answer_text.text = btn.text

        if  answer_text.text != "": 
            root = App.get_running_app().root
            print("answer: ", answer_text.text)
            if answer_text.text == "Yes":
                root.hmi_popup.open(True)
            else:
                root.hmi_popup.open(False)
            # Clear the answer text
            answer_text.text = ""

################################################################################
class KivyTutorApp(App):
    
    #App object
    def __init__(self, **kwargs):
        super(KivyTutorApp, self).__init__(**kwargs)
        self.use_kivy_settings = False
        Window.bind(on_keyboard=self.onBackBtn)

    def onBackBtn(self, window, key, *args):
        # user presses back button
        if key == 27:
            return self.root.onBackBtn()

    def build(self):
        return KivyTutorRoot()

    def getText(self):
        return ("Hey There!\nThis App was built using"
                "[b][ref=kivy]kivy[/ref][/b]\n"
                "Feel free to look at the source code "
                "[b][ref=source]here[/ref][/b].\n"
                "This app is under the [b][ref=mit]MIT License[/ref][/b]\n"
                "My site: [b][ref=website]PyGopar.com[/ref][/b]")

    def on_ref_press(self, instance, ref):
        _dict = {
            "source": "https://github.com/gopar/Kivy-Tutor",
            "website": "http://www.pygopar.com",
            "kivy": "http://kivy.org/#home",
            "mit": "https://github.com/gopar/Kivy-Tutor/blob/master/LICENSE"
        }

        webbrowser.open(_dict[ref])

    def build_config(self, config):
        config.setdefaults("General", {"lower_num": 0, "upper_num": 10})

    def build_settings(self, settings):
        settings.add_json_panel("Kivy Hmi Tutor", self.config,
                                data=json_settings)

    def on_config_change(self, config, section, key, value):
        if key == "upper_num":
            self.root.hmi_screen.max_num = int(value)
        elif key == "lower_num":
            self.root.hmi_screen.min_num = int(value)

if __name__ == '__main__':
    KivyTutorApp().run()
