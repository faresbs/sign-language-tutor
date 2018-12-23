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

#For music 
from kivy.core.audio import SoundLoader

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

#NO NEED FOR NOW
#For Gifs
class MyImage(Image):
        frame_counter = 0
        frame_number = 7 # depends on the gif frames

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

        #Dialogue to interact with the user
        self.path_diaologue = 'dialogue'
        self.dialogues = os.listdir(self.path_diaologue)

        self.init_dialogue = True

        #Check if there is a next dialogue in the current state
        self.dialogue_idx  = 0



    def changeScreen(self, next_screen):

        # If screen is not already in the list fo prevous screens
        if self.ids.kivy_screen_manager.current not in self.screen_list:
            self.screen_list.append(self.ids.kivy_screen_manager.current)

        if next_screen == "about this app":
            self.ids.kivy_screen_manager.current = "about_screen"

        else:
           
            #Is the states order in the alphabetical order or not, depends on mode
            if (next_screen == 'challenge'):
                shuffle(self.states)
                self.init_state = self.states[0]
                self.hmi_screen.question_image.text = self.states[0]

                #Reset
                self.hmi_screen.button.idx = 0
                self.score = 0
                self.dialogue_idx  = 0
                self.current = 0

                #When you change from modes
                if(self.init_dialogue == False):
                    self.hmi_screen.interact_button.text = ""


            if (next_screen == 'learn'):
                self.states.sort()
                self.init_state = self.states[0]
                self.hmi_screen.question_image.text = self.states[0]

                #Reset
                self.hmi_screen.button.idx = 0
                self.score = 0
                self.dialogue_idx  = 0
                self.current = 0

                #When you change from modes
                if(self.init_dialogue == False):
                    self.hmi_screen.interact_button.text = ""



            #image = self.path+'/'+self.init_state+'/image.png'
            image = self.path+'/'+self.init_state+"/gif.gif"
            self.hmi_screen.image.source = image
            self.ids.kivy_screen_manager.current = "hmi_screen"
    


    def changeState(self):


        if (len(self.states) == 0):
            print ("There are no states!")
            sys.quit()


        idx = self.hmi_screen.button.idx 
        self.current = idx

        if idx < len(self.states):
            letter = self.states[idx]

            #image = self.path+'/'+letter+'/image.png'
            #image = self.myImage(source = self.path+'/'+letter+"/gif.gif")
            image = self.path+'/'+letter+"/gif.gif"
            
            self.hmi_screen.image.source = image    
            self.hmi_screen.question_image.text = letter         
            self.hmi_screen.button.idx += 1  
            #Reset the dialogue counter
            self.dialogue_idx = 0
            #Go the next dialogue state letter

        #Here if we exceed the len of states, we are done => finish == True 
        else:
            self.hmi_popup.open('Done', self.score)
            print('done!')

            #Reset loop
            #self.current = 0
            self.hmi_screen.button.idx = 0

    

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
            cv2.imshow('Point you hand at me :)',frame)
            
            #Call predictor
            prediction = rec.predict(frame, class_model, detect_model, args, class_names)

            count += 1

            # HOW TO SAFELY CLOSE THE CAM WINDOW using a button not q
            if cv2.waitKey(1) & 0xFF == ord('q'):#to get out from the infinite loop
                break

            if (prediction == self.list[self.current]):

                #Go to the next letter
                #self.current += 1
                #increase score
                self.score += 100

                self.hmi_popup.open('Yes', self.score)

                break

            #If exceed count limit then display error message
            if(count >= 50):
                #decrease score
                self.score -= 10
                self.hmi_popup.open('No', self.score)

                break

        #Destroy window cam
        cap.release()
        cv2.destroyAllWindows()


    #Instructive and interactive Dialogue with the user 
    #A button when you click on it, it displays the next message
    def interaction(self):

        #print(self.dialogue_idx)
        #print(self.states[self.current])

        if (len(self.dialogues) == 0):
            print ("There are no dialogues!")
            sys.quit()
        
        #Read current dialogue using the states[idx] from dialogue folder
        if(self.init_dialogue == True):

            #Get the init dialogue file
            #Starts only one time during the app launch
            file = self.path_diaologue+"/init.txt"
        else:
            #Get the current state letter dialogue file
            file = self.path_diaologue+'/'+self.states[self.current]+'.txt'


        #Save lines of current states in a list
        with open(file) as f:
            list_dialogue = f.read().splitlines()

        #print(list_dialogue)

        if(self.dialogue_idx < len(list_dialogue)):
            self.hmi_screen.interact_button.text = list_dialogue[self.dialogue_idx]
            self.dialogue_idx += 1
        else:
            if(self.init_dialogue == True):
                #Go to the next dialogue states
                #init dialogue is done just one time during the app launch 
                self.init_dialogue = False
                
            self.hmi_screen.interact_button.text = "Click to show description!"


        
        #self.hmi_screen.interact_button.text = "  Put your fingers all straight up and touching and then bend your "



################################################################################
class HmiScreen(Screen, Arithmetic):
    
    #Widget that will arc as a screen and hold funcs for hmi questions
    def __init__(self, *args, **kwargs):
        super(HmiScreen, self).__init__(*args, **kwargs)

################################################################################
class HmiPopup(Popup):
    
    #Popup for telling user whether he got it right or wrong
    GOOD = "{}\n:)"
    BAD = "{}\nTry again!!"

    #good_index = 0
    bad_index = -1

    #Read response messages from txt files
    with open('good_response.txt') as f:
        GOOD_LIST = f.read().splitlines()

    with open('bad_response.txt') as f:
        BAD_LIST = f.read().splitlines()

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

        if(self.bad_index >= len(self.BAD_LIST)-1):
            self.bad_index = -1

        if answer == 'Yes':
            index = random.randint(0, len(self.GOOD_LIST) - 1)
            return self.GOOD.format(self.GOOD_LIST[index])
        elif answer == 'No':
            #Dont do random
            hmi_screen = App.get_running_app().root.hmi_screen
            self.bad_index += 1

            return self.BAD.format(self.BAD_LIST[self.bad_index])

        elif answer== 'Done':
            if(score > 0):
                return 'You did it, GOOD JOB!\n'+'Score: '+str(score)
            else:
                return 'Maybe if you try again, you\'ll get them all this time :)\nClick on "Next" to reset.'
            
            
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

        #Add background music
        sound = SoundLoader.load('periwinkle.mp3')
        if sound:
            print("Sound found at %s" % sound.source)
            print("Sound is %.3f seconds" % sound.length)
            
            #loop the background music
            sound.loop = True
            #start with 50% volume
            sound.volume = 0.5

            sound.play()

    def onBackBtn(self, window, key, *args):
        # user presses back button
        if key == 27:
            return self.root.onBackBtn()

    def build(self):

        #TO DO: SETTINGS MUISC : VOLUME + MUTE

        

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
        config.setdefaults("General", {"volume_music": 1, "mute_music": False})

    def build_settings(self, settings):
        settings.add_json_panel("Kivy Hmi Tutor", self.config,
                                data=json_settings)

    def on_config_change(self, config, section, key, value):
        if key == "volume_music":
            self.root.hmi_screen.volume = int(value)

        elif key == "mute_music":
            self.root.hmi_screen.max_num = int(value)


if __name__ == '__main__':
    KivyTutorApp().run()
