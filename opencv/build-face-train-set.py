#!/usr/bin/env python

import sys
from PIL import Image, ImageTk
import csv
import time
import Tkinter
import os

"""
expects to be passed a find-style list of jpegs;
show images and store responses in csv
.... dependencies:
  dnf install tkinter tk-devel tcl-devel
  pip install Pillow
.... help with window stuff:
http://code.activestate.com/recipes/521918-pil-and-tkinter-to-display-images/
"""

if len(sys.argv) == 1:
    print("No input file specified")
    sys.exit(2)
image_list = sys.argv[1]
csv_fieldnames = ['image', 'num_frontal_faces']
csv_filename = 'results.csv'
current_image = ''

def button_click_exit_mainloop(event):
    event.widget.quit() # this will cause mainloop to unblock.

def save_input(event, entry):
    global csv_writer
    the_input = entry.get()
    print(the_input)
    csv_writer.writerow([current_image,the_input])
    entry.delete(0,Tkinter.END)
    event.widget.quit() 

root = Tkinter.Tk()
input_box = Tkinter.Toplevel()
text = Tkinter.Text(input_box)
text.pack()
text.insert(Tkinter.CURRENT,"Enter number of frontal faces")
e = Tkinter.Entry(input_box)
e.pack()

# the lambda allows passing additional args to callback:
input_box.bind("<Return>", lambda event, arg=e: save_input(event, arg))
input_box.bind("<KP_Enter>", lambda event, arg=e: save_input(event, arg))

root.geometry('+%d+%d' % (100,100))
old_label_image = None

with open(csv_filename, 'ab') as csv_file:
    csv_writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
    # write headers if file is empty:
    if os.stat(csv_filename).st_size == 0:
        csv_writer.writerow(csv_fieldnames)
    with open(image_list) as g:
        for f in g:
            try:
                f = f.rstrip()
                current_image = f
                image = Image.open(f)
                width, height = image.size
                width = width/4
                height = height/4
                image = image.resize((width,height), Image.ANTIALIAS)
                root.geometry('%dx%d' % (width,height))
                tkpi = ImageTk.PhotoImage(image)
                label_image = Tkinter.Label(root, image=tkpi)
                label_image.place(x=0,y=0,width=width,height=height)
                root.title(f)
                if old_label_image is not None:
                    old_label_image.destroy()
                old_label_image = label_image
                root.mainloop() # wait until user clicks the window
            except Exception, e:
                print("EXCEPTION: %s" % e)
                sys.exit(1)
