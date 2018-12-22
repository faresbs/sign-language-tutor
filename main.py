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
import os
import sys
from random import shuffle
import time

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

#For Gifs
class MyImage(Image):
        frame_counter = 0
        frame_number = 7 # my example GIF had 2 frames

        def on_texture(self, instance, value):     
            if self.frame_counter == self.frame_number + 1:
                self._coreimage.anim_reset(False)
            self.frame_counter += 1
     

class KivyTutorRoot(BoxLayout):
    """
    Root of all widgets
    """
    hmi_screen = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(KivyTutorRoot, self).__init__(**kwargs)
        # List of previous screens
        self.screen_list = []
        
        #Is the states order in the alphabetical order or not
        #NO NEED
        self.is_mix = False
        
        self.hmi_popup = HmiPopup()
        #self.myImage = MyImage(Image)

        self.current = 0
        
        #To campare with prediction result
        self.list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']
        
        self.score = 0

        #Recover alphabet states: (folders where each has the corresponding letter and image/gif)
        self.path = 'states'
        self.states = os.listdir(self.path)

        self.init_state = 'A'



    def changeScreen(self, next_screen):

        # If screen is not already in the list fo prevous screens
        if self.ids.kivy_screen_manager.current not in self.screen_list:
            self.screen_list.append(self.ids.kivy_screen_manager.current)

        if next_screen == "about this app":
            self.ids.kivy_screen_manager.current = "about_screen"

        else:
           
            #Is the states order in the alphabetical order or not, depends on the difficulty

            if (next_screen == 'challenge'):
                shuffle(self.states)
                self.init_state = self.states[0]
                self.hmi_screen.question_image.text = self.states[0]

                #Reset
                self.current = 0
                self.hmi_screen.button.idx = 0
                self.score = 0

            if (next_screen == 'learn'):
                self.states.sort()
                self.init_state = self.states[0]
                self.hmi_screen.question_image.text = self.states[0]

                #Reset
                self.current = 0
                self.hmi_screen.button.idx = 0
                self.score = 0


            image = self.path+'/'+self.init_state+'/image.png'
            self.hmi_screen.image.source = image
            self.ids.kivy_screen_manager.current = "hmi_screen"
    


    def changeScreen_(self):


        if (len(self.states) == 0):
            print ("There are no folders!")
            sys.quit()


        idx = self.hmi_screen.button.idx 
        print(idx)

        if idx < len(self.states):
            letter = self.states[idx]

            #image = self.path+'/'+letter+'/image.png'
            #image = self.myImage(source = self.path+'/'+letter+"/gif.gif")
            image = self.path+'/'+letter+"/gif.gif"
            
            self.hmi_screen.image.source = image    
            self.hmi_screen.question_image.text = letter         
            self.hmi_screen.button.idx += 1  

        #Here if we exceed the len of states, we are done => finish == True 
        else:
            self.hmi_popup.open('Done', self.score)
            print('done!')

            #Reset loop
            self.current = 0
            self.hmi_screen.button.idx = 0


        #This is to make it loop
        #else:
        #    self.hmi_screen.button.idx = 0
        #    idx = self.hmi_screen.button.idx
        #    letter = self.states[idx]
        #    image = self.path+'/'+letter+'/image.png'
        #    self.hmi_screen.image.source = image
        #    self.hmi_screen.question_image.text = letter 
        #    self.hmi_screen.button.idx = 1
    

    def onBackBtn(self):
        # Check if there are any screen to go back to
        if self.screen_list:
            # if there are screens we can go back to, the just do it
            self.ids.kivy_screen_manager.current = self.screen_list.pop()
            # Saw we don't want to close
            return True
        # No more screens to go back to
        return False

    
    def camCapture(self):

        count = 0

        cap = cv2.VideoCapture(0)
        #fourcc = cv2.VideoWriter_fourcc(*'XVID') #pour enregistrer lw fichier codec

        #load recognition models just one time
        class_model, detect_model, args, class_names = rec.load_models()

        while(True):
            ret, frame = cap.read()
            cv2.imshow('frame',frame)
            
            #Call predictor
            prediction = rec.predict(frame, class_model, detect_model, args, class_names)

            count += 1

            # HOW TO SAFELY CLOSE THE CAM WINDOW using a button not q
            if cv2.waitKey(1) & 0xFF == ord('q'):#to get out from the infinite loop
                break

            if (prediction == self.list[self.current]):

                self.current += 1
                #increase score
                self.score += 100
                self.hmi_popup.open('Yes', self.score)

                break

            #If exceed count limit then display error message
            if(count >= 200):
                #decrease score
                self.score -= 10
                self.hmi_popup.open('No', self.score)

                break

        #Destroy window cam
        cap.release()
        cv2.destroyAllWindows()


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
    
    def open(self, answer, score):
        # If answer is correct take off button if its visible
        if answer == 'Yes':
            if self.wrapped_button in self.content.children:
                self.content.remove_widget(self.wrapped_button)
        # If answers is wrong, display button if not visible
        elif answer == 'No':
            if self.wrapped_button not in self.content.children:
                self.content.add_widget(self.wrapped_button)
        elif answer == 'Done':
            if self.wrapped_button not in self.content.children:
                self.content.add_widget(self.wrapped_button)


        # Set up text message
        self.message.text = self._prep_text(answer, score)

        # display popup
        super(HmiPopup, self).open()
        if answer == 'Yes':
            #pop up vanish after n sec 
            Clock.schedule_once(self.dismiss, 10)

    def _prep_text(self, answer, score):
        if answer == 'Yes':
            index = random.randint(0, len(self.GOOD_LIST) - 1)
            return self.GOOD.format(self.GOOD_LIST[index])
        elif answer == 'No':
            #Dont do random : Almost -> try again -> you got this -> try with the next letter
            index = random.randint(0, len(self.BAD_LIST) - 1)
            hmi_screen = App.get_running_app().root.hmi_screen
            return self.BAD.format(self.BAD_LIST[index])

        elif answer== 'Done':
            print(score)
            if(score > 0):
                return 'You did them all, GOOD JOB!\n'+'Score: '+str(score)
            else:
                return 'Maybe if you try again, you get them all right this time :)'
            
            
################################################################################
class KeyPad(GridLayout):
    
    #Documentation for KeyPad
    def __init__(self, *args, **kwargs):
        super(KeyPad, self).__init__(*args, **kwargs)
        self.cols = 3
        self.spacing = 10
        #self.createButtons()

    #def createButtons(self):
        #_list = ["Yes", "No", "Next!"]
        #for num in _list:
            #self.add_widget(Button(text=str(num), on_release=self.onBtnPress))
                       
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
                root.hmi_popup.open('Yes')
            elif answer_text.text == "No":
                root.hmi_popup.open('No')
            else:
                root.hmi_popup.open("Done")

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

# HOW TO RESET THE APP
def reset():
    import kivy.core.window as window
    from kivy.base import EventLoop
    
    if not EventLoop.event_listeners:
        from kivy.cache import Cache
        
    window.Window = window.core_select_lib('window', window.window_impl, True)
    Cache.print_usage()
        
    for cat in Cache._categories:
        Cache._objects[cat] = {}


if __name__ == '__main__':
    KivyTutorApp().run()
