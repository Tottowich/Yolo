# import tkinter as tk
# from PIL import Image, ImageTk
# import numpy as np
# import time

# class GUIWithImages:
#     def __init__(self, root:tk.Tk=None, tiles:tuple[int]=(1,2), titles:tuple[str]=("Image 1", "Image 2")):
#         self.root = root
#         self.image_labels = [[None for _ in range(tiles[1])] for _ in range(tiles[0])]
#         self.titles = titles
#     def create_gui(self):
#         self.root.title("Image Viewer")

#         frame = tk.Frame(self.root)
#         frame.pack(fill="both", expand=True)
#         # Tiled version!
#         for i in range(len(self.image_labels)):
#             for j in range(len(self.image_labels[i])):
#                 image_frame = tk.Frame(frame)
#                 image_frame.grid(row=i, column=j, sticky="nsew")
#                 self.image_labels[i][j] = tk.Label(image_frame, anchor="nw")
#                 self.image_labels[i][j].pack(fill="both", expand=True)
#                 # Set title above image
#                 tk.Label(image_frame, text=self.titles[i*len(self.image_labels[i])+j]).pack(fill="both", expand=True)


#     def update_image(self, image, row, col):

#         self.show_image(image, row, col)
#         self.root.update()

#     def show_image(self, image, row, col):
#         if image is not None:
#             pil_image = Image.fromarray(image)
#             photo = ImageTk.PhotoImage(pil_image)
#             self.image_labels[row][col].configure(image=photo)
#             self.image_labels[row][col].image = photo

#     def run(self):
#         self.root.mainloop()

# def generate_dummy_image(size=(100, 100, 3)):
#     image = np.random.randint(0, 256, size, dtype=np.uint8)
#     return image

# # ... your code ...
# if __name__ == "__main__":
#     # Create GUI
#     gui = GUIWithImages(tk.Tk(), (2,2), titles=("Image 1", "Image 2", "Image 3", "Image 4"))
#     gui.create_gui()
#     # Run the GUI
#     # Update the images
#     for i in range(1000):
#         gui.update_image(generate_dummy_image(),0,0)
#         gui.update_image(generate_dummy_image(),0,1)
#         gui.update_image(generate_dummy_image(),1,0)
#         gui.update_image(generate_dummy_image(),1,1)
#         gui.root.update()
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np

class GUIWithImages:
    def __init__(self, root:tk.Tk=None, tiles:tuple[int]=(1,2), titles:tuple[str]=("Image 1", "Image 2")):
        if root is not None:
            self.root = root
        else:
            self.root = tk.Tk()
            self.root.style = ttk.Style()
            self.root.style.theme_use("clam")
        self.image_labels = [[None for _ in range(tiles[1])] for _ in range(tiles[0])]
        self.titles = titles

    def create_gui(self):
        self.root.title("Image Viewer")

        frame = ttk.Frame(self.root)
        frame.pack(fill="both", expand=True)

        for i in range(len(self.image_labels)):
            for j in range(len(self.image_labels[i])):
                image_frame = ttk.Frame(frame)
                image_frame.grid(row=i, column=j, padx=10, pady=10, sticky="nsew")

                label_frame = ttk.Frame(image_frame)
                label_frame.pack(side="top", fill="x")

                title_label = ttk.Label(label_frame, text=self.titles[i*len(self.image_labels[i])+j], font=("Arial", 12, "bold"))
                title_label.pack(side="top", pady=5)

                self.image_labels[i][j] = ttk.Label(image_frame)
                self.image_labels[i][j].pack(fill="both", expand=True)

    def update_image(self, image, row, col):
        self.show_image(image, row, col)

    def show_image(self, image, row, col):
        if image is not None:
            pil_image = Image.fromarray(image)
            photo = ImageTk.PhotoImage(pil_image)
            self.image_labels[row][col].configure(image=photo)
            self.image_labels[row][col].image = photo

    def run(self):
        self.root.mainloop()

def generate_dummy_image(size=(100, 100, 3)):
    image = np.random.randint(0, 256, size, dtype=np.uint8)
    return image

# ... your code ...
if __name__ == "__main__":
    # Create GUI
    root = tk.Tk()
    root.style = ttk.Style()
    root.style.theme_use("nightblue")

    gui = GUIWithImages(root, (2,2), titles=("Image 1", "Image 2", "Image 3", "Image 4"))
    gui.create_gui()

    # Run the GUI
    # Update the images
    for i in range(1000):
        gui.update_image(generate_dummy_image(), 0, 0)
        gui.update_image(generate_dummy_image(), 0, 1)
        gui.update_image(generate_dummy_image(), 1, 0)
        gui.update_image(generate_dummy_image(), 1, 1)
        root.update()
