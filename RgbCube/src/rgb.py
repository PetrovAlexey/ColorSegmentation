import ComplexKmeans

import cv2
import numpy as np
import pptk
import pandas as pd

from sklearn.cluster import KMeans

from tkinter import *
from tkinter import filedialog

from tkinter import ttk

from PIL import Image, ImageTk

import matplotlib
import matplotlib.pyplot as plt
import pptk


class Paint(Frame):
    def initial_points(self):
        self.start_x = -1
        self.start_y = -1
        self.finish_x = -1
        self.finish_y = -1

    def __init__(self, parent):
        ttk.Frame.__init__(self, parent)
        self.initial_points()
        self.polygone = []
        self.line = None
        self.crop_active = False
        self.img = None
        self.image = None
        self.image_path = None

        self.parent = parent
        self.color = "black"
        self.brush_size = 2
        self.setUI()
        self._draw_image()

        self.ph = None
        self.shape = None
        
        # Simple KMeans config
        self._num_clusters = 4
        
        # Complex Kmeans config
        self.n_clusters = 2
        self.n_lines = 6
        self.n_init = 6
        self.max_iter = 8
        

    def rgb_cube(self):
        X = [[0, 0, i, j, k] for i in range(0, 255)
             for j in {0, 255} for k in {0, 255}]
        Y = [[0, 0, j, i, k] for i in range(0, 255)
             for j in {0, 255} for k in {0, 255}]
        Z = [[0, 0, j, k, i] for i in range(0, 255)
             for j in {0, 255} for k in {0, 255}]
        return np.vstack((X, Y, Z))

    def set_crop(self, status):
        self.crop_active = status

        if (status is True):
            self.polygone = []

    def set_color(self, new_color):
        self.color = new_color

    def set_brush_size(self, new_size):
        self.brush_size = new_size

    def _draw_image(self):
        if (self.image is None):
            return
        self.canv.create_image(0, 0, anchor="nw", image=self.image)

    def draw(self, event):
        if (self.crop_active is False):
            return

        self.polygone.append((event.x, event.y))
        if (self.start_x != -1 or self.start_y != -1):
            self.finish_x = self.start_x
            self.finish_y = self.start_y

        self.start_x = event.x
        self.start_y = event.y

        if not self.line:
            self.line = self.canv.create_line(
                self.start_x,
                self.start_y,
                event.x,
                event.y,
                dash=(4, 2),
                fill=self.color,
                width=5)

        self.canv.create_oval(event.x - self.brush_size,
                              event.y - self.brush_size,
                              event.x + self.brush_size,
                              event.y + self.brush_size,
                              fill=self.color, outline=self.color)

    def draw_line(self, event):
        self.crop(event)
        if (self.crop_active is False):
            return

        if (self.start_x == -1 or self.start_y == -1):
            return
        self.canv.coords(self.line,
                         self.start_x,
                         self.start_y,
                         event.x,
                         event.y)

    def on_button_release(self, event):
        if (self.crop_active is False):
            return

        if (self.finish_x != -1 or self.finish_y != -1):
            self.canv.create_line(self.start_x,
                                  self.start_y,
                                  self.finish_x,
                                  self.finish_y,
                                  dash=(4, 2),
                                  fill=self.color,
                                  width=5)

    def on_right_button_press(self, event):
        self.initial_points()
        self.set_crop(False)
        self.canv.coords(self.line, 0, 0, 0, 0)
        self.line = None

    def print_polygone(self):
        if (len(self.polygone) == 0):
            return
        print(self.polygone)
        print(self.image_path)
        image = cv2.cvtColor(np.array(self.img_resize), cv2.COLOR_RGB2BGR)

        mask = np.zeros(image.shape, dtype=np.uint8)

        roi_corners = np.array([self.polygone], dtype=np.int32)
        channel_count = image.shape[2]
        ignore_mask_color = (255,)*channel_count
        cv2.fillPoly(mask, roi_corners, ignore_mask_color)

        result = cv2.bitwise_and(image, mask)
        result[mask == 0] = 255

        cv2.imshow('result', result)
        cv2.waitKey()

        colourImg = Image.fromarray(result)

        colourPixels = colourImg.convert("RGB")
        colourArray = np.array(colourPixels.getdata()).reshape(
            colourImg.size + (3,))
        indicesArray = np.moveaxis(np.indices(colourImg.size), 0, 2)
        imgArray = np.dstack((indicesArray, colourArray)).reshape((-1, 5))
        # add rgb cube
        self.allArray = np.vstack((imgArray, self.rgb_cube()))

    def draw_cluster(self):
        mean_df = self.df.groupby('y_kmeans')['red', 'green', 'blue'].mean()
        
        for cluster in self.set_colours:
            try:
                self.df.loc[self.df['y_kmeans'] == cluster, 'red_cluster'] = mean_df.loc[cluster][0]
                self.df.loc[self.df['y_kmeans'] == cluster, 'green_cluster'] = mean_df.loc[cluster][1]
                self.df.loc[self.df['y_kmeans'] == cluster, 'blue_cluster'] = mean_df.loc[cluster][2]
            except:
                print(f'Cluster {cluster} is empty')
                continue
        
        self.draw_df(self.df[['red', 'green', 'blue']], self.df[['red_cluster', 'green_cluster', 'blue_cluster']] / 255)

    def draw_df(self, df, colour):
        v = pptk.viewer(df, colour)
        v.set(point_size=0.3)

    def draw_rgb_poly(self):
        df = pd.DataFrame(self.allArray,
                          columns=["y", "x", "blue", "green", "red"])
        self.draw_df(df[['red', 'green', 'blue']], df[['red', 'green', 'blue']] / 255.)

    def draw_rgb(self):
        df = self.generate_df(self.img, True)
        self.draw_df(df[['red', 'green', 'blue']], df[['red', 'green', 'blue']] / 255.)
        
    def generate_picture(self):
        mean_df = self.df.groupby('y_kmeans')['red', 'green', 'blue'].mean()
        print(mean_df)
        for cluster in self.set_colours:
            try:
                self.df.loc[self.df['y_kmeans'] == cluster, 'red_cluster'] = mean_df.loc[cluster][0]
                self.df.loc[self.df['y_kmeans'] == cluster, 'green_cluster'] = mean_df.loc[cluster][1]
                self.df.loc[self.df['y_kmeans'] == cluster, 'blue_cluster'] = mean_df.loc[cluster][2]
            except e:
                #print(f'Cluster {cluster} is empty')
                print(e)
                continue
        test = (self.df[['red_cluster','green_cluster', 'blue_cluster']]).to_numpy().reshape(self.image_path.size[::-1] + (3, ))
        return test

    

    def cluster_complex(self):
        #if "y_kmeans" not in self.df:
        X = self.df[['red', 'green', 'blue']].to_numpy()
        kmean_test = ComplexKmeans.ComplexKmeans(n_clusters=self.n_clusters, n_lines=self.n_lines, max_iter=self.max_iter, n_init=self.n_init)
        [colours, new_planes, new_centroids, array_lines, array_centroids] = kmean_test.fit(X)
        self.df["y_kmeans"] = colours
        self.set_colours = set(colours)
            
        self.test = self.generate_picture()
        self.ph = ImageTk.PhotoImage(image=Image.fromarray((self.test).astype(np.uint8), mode='RGB'))
        self.canv_cluster.create_image(0, 0, anchor="nw", image=self.ph)
        
        
    def cluster(self):
        #if "y_kmeans" not in self.df:
        kmeans = KMeans(n_clusters=self._num_clusters)
        kmeans.fit(self.df[['red', 'green', 'blue']])
        y_kmeans = kmeans.predict(self.df[['red', 'green', 'blue']])
        self.df["y_kmeans"] = y_kmeans
        self.set_colours = set(y_kmeans)
        self.test = self.generate_picture()
        self.ph = ImageTk.PhotoImage(image=Image.fromarray((self.test).astype(np.uint8), mode='RGB'))
        self.canv_cluster.create_image(0, 0, anchor="nw", image=self.ph)
        

    def openfn(self):
        filename = filedialog.askopenfilename(title='open')
        return filename

    def generate_df(self, img, rgb):
        colourPixels = img.convert("RGB")
        colourArray = np.array(colourPixels.getdata()).reshape(
                img.size + (3,))
        indicesArray = np.moveaxis(np.indices(img.size), 0, 2)
        imgArray = np.dstack((indicesArray, colourArray)).reshape((-1, 5))
        if rgb:
            allArray = np.vstack((imgArray, self.rgb_cube()))
        else:
            allArray = imgArray

        return pd.DataFrame(allArray,
                            columns=["y", "x", "red", "green", "blue"])

    def open_image(self):
        self.canv.delete("all")
        try:
            path = self.openfn()
            self.img = Image.open(path)
            self.image_path = self.img
            new_width, new_height = self.img.size
            self.img_resize = self.img.copy()
            self.image = ImageTk.PhotoImage(self.img)
            
            img_width, img_height = self.img.size
            scale_w = self.canv.winfo_width() / img_width
            scale_h = self.canv.winfo_height() / img_height
            scale = min(scale_w, scale_h)
            copy_of_image = self.img.copy()
            image = copy_of_image.resize((int(img_width*scale), int(img_height*scale)))
            self.img_resize = image.copy()
            self.image = ImageTk.PhotoImage(image)
                        
            self._draw_image()
            self.polygone = []
            self.ph = None
            self.df = self.generate_df(self.img, False)
        except e:
            print("Could not open file" + e)

    def setUI(self):
        self.parent.title("RGB Cube")
        self.pack(expand=1, fill=BOTH)

        self.columnconfigure(7, weight=1)
        self.columnconfigure(16, weight=1)
        self.rowconfigure(2, weight=1)

        self.canv = Canvas(self, bg="white", cursor="cross")
        self.canv.grid(row=2, column=0, columnspan=8,
                       padx=5, pady=5, sticky=E+W+S+N)

        self.canv_cluster = Canvas(self, bg="white", cursor="cross")
        self.canv_cluster.grid(row=2, column=9, columnspan=8,
                               padx=5, pady=5, sticky=E+W+S+N)

        self.canv.bind("<ButtonPress-1>", self.draw)
        self.canv.bind("<ButtonPress-3>", self.on_right_button_press)
        self.canv.bind("<Motion>", self.draw_line)
        self.canv.bind("<ButtonRelease-1>", self.on_button_release)

        color_lab = ttk.Label(self, text="Image: ")
        color_lab.grid(row=0, column=0, padx=6)

        red_btn = ttk.Button(self, text="Open", width=10,
                             command=self.open_image)
        red_btn.grid(row=0, column=1)

        green_btn = ttk.Button(self, text="Crop", width=10,
                               command=lambda: self.set_crop(True))
        green_btn.grid(row=0, column=2)

        clear_btn = ttk.Button(self, text="Clear all", width=10,
                               command=lambda: self.canv.delete("all"))
        clear_btn.grid(row=0, column=6, sticky=W)

        size_lab = ttk.Label(self, text="Draw: ")
        size_lab.grid(row=1, column=0, padx=5)
        one_btn = ttk.Button(self, text="RGB", width=10,
                             command=self.draw_rgb)
        one_btn.grid(row=1, column=1)

        two_btn = ttk.Button(self, text="RGB Poly", width=10,
                             command=self.draw_rgb_poly)
        two_btn.grid(row=1, column=2)

        twenty_btn = ttk.Button(self, text="Draw Poly", width=10,
                                command=self.print_polygone)
        twenty_btn.grid(row=1, column=6, sticky=W)

        draw_cluster_btn = ttk.Button(self, text="Draw Cluster", width=15,
                                      command=self.draw_cluster)
        draw_cluster_btn.grid(row=1, column=7, sticky=W)

        cluster_btn = ttk.Button(self, text="Cluster", width=10,
                                 command=self.cluster)
        cluster_btn.grid(row=0, column=6, sticky=W)
        
        complex_cluster_btn = ttk.Button(self, text="Complex Cluster", width=15,
                                 command=self.cluster_complex)
        complex_cluster_btn.grid(row=0, column=7, sticky=W)

        #self.comboExample = ttk.Combobox(self,
        #                                 values=[0, 1, 2, 3])
        #self.comboExample.grid(row=1, column=7)
        #self.comboExample.current(0)

        self.zoomcycle = 0
        self.zimg_id = None

        self.canv.bind("<MouseWheel>", self.zoomer)

        self.canv.bind('<Configure>', self.resize_image)

    def zoomer(self, event):
        if (event.delta > 0):
            if self.zoomcycle != 4:
                self.zoomcycle += 1
        elif (event.delta < 0):
            if self.zoomcycle != 0:
                self.zoomcycle -= 1
        self.crop(event)

    def crop(self, event):
        if self.zimg_id:
            self.canv.delete(self.zimg_id)
        if (self.zoomcycle) != 0:
            x, y = event.x, event.y
            if self.zoomcycle == 1:
                tmp = self.img_resize.crop((x - 45, y - 30, x + 45, y + 30))
            elif self.zoomcycle == 2:
                tmp = self.img_resize.crop((x - 30, y - 20, x + 30, y + 20))
            elif self.zoomcycle == 3:
                tmp = self.img_resize.crop((x - 15, y - 10, x + 15, y + 10))
            elif self.zoomcycle == 4:
                tmp = self.img_resize.crop((x - 6, y - 4, x + 6, y + 4))
            size = 300, 200
            self.zimg = ImageTk.PhotoImage(tmp.resize(size))
            self.zimg_id = self.canv.create_image(event.x, event.y, image=self.zimg)

    def resize_image(self, event):
        if self.img is None:
            return
        new_width = event.width
        new_height = event.height
        img_width, img_height = self.img.size
        scale_w = new_width / img_width
        scale_h = new_height / img_height
        scale = min(scale_w, scale_h)
        copy_of_image = self.img.copy()
        image = copy_of_image.resize((int(img_width*scale), int(img_height*scale)))
        self.img_resize = image.copy()
        self.image = image.copy()
        photo = ImageTk.PhotoImage(image)
        self.canv.create_image(0, 0, anchor="nw", image=photo)
        self.canv.image = photo
        # avoid garbage collection
        if "y_kmeans" not in self.df:
            return
        copy_of_cluster = Image.fromarray((self.test).astype(np.uint8), mode='RGB').copy()
        self.ph_resize = copy_of_cluster.resize((int(img_width*scale), int(img_height*scale)))
        self.ph = self.ph_resize.copy()
        cluster = ImageTk.PhotoImage(self.ph_resize)
        self.canv_cluster.create_image(0, 0, anchor="nw", image=cluster)
        self.canv_cluster.image = cluster
        
        # avoid garbage collection


def main():
    root = Tk()
    root.geometry("850x500+300+300")
    app = Paint(root)
    root.mainloop()


if __name__ == '__main__':
    main()
