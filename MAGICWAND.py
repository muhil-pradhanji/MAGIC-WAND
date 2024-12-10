import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import simpledialog

def flood_fill(img, seed_point, thresh):
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), np.uint8)
    flood_flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE
    num, im, mask, rect = cv2.floodFill(img, mask, seed_point, 255, (thresh,) * 3, (thresh,) * 3, flood_flags)
    return mask[1:-1, 1:-1]

def find_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

class MagicWandTool:
    def __init__(self, image_path, destination_image_path):
        self.img = self.load_image(image_path)
        self.original_img = self.img.copy()
        self.selected_contours = []
        self.threshold = 10  # Initial tolerance value
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.img)
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('scroll_event', self.onscroll)
        #self.fig.canvas.mpl_connect('motion_notify_event', self.onhover)
        self.zoom_scale = 1.2  
        self.output_dir = r"C:\muhil\output1"
        os.makedirs(self.output_dir, exist_ok=True)
        self.destination_image = self.load_image(destination_image_path)
        self.dest_fig, self.dest_ax = plt.subplots()
        self.dest_ax.imshow(self.destination_image)
        self.dest_fig.canvas.mpl_connect('button_press_event', self.on_dest_click)
        plt.show(block=False)  # Display the image first
        self.show_tolerance_dialog()

    def load_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image at {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def show_tolerance_dialog(self):
        self.root = tk.Tk()
        self.root.title("Adjust Tolerance")

        self.tolerance_value = tk.IntVar()
        self.tolerance_value.set(self.threshold)

        # Create and pack widgets
        self.label = tk.Label(self.root, text="Set Tolerance:")
        self.label.pack(pady=10)

        self.slider = tk.Scale(self.root, from_=0, to=255, orient='horizontal', variable=self.tolerance_value, length=400, sliderlength=40, tickinterval=25)
        self.slider.pack(pady=10)

        self.tolerance_display = tk.Label(self.root, text=f"Tolerance: {self.threshold}")
        self.tolerance_display.pack(pady=10)

        self.slider.bind("<Motion>", self.update_tolerance_display)

        self.entry_label = tk.Label(self.root, text="Enter Tolerance:")
        self.entry_label.pack(pady=5)

        self.tolerance_entry = tk.Entry(self.root)
        self.tolerance_entry.insert(0, str(self.threshold))
        self.tolerance_entry.pack(pady=5)

        self.apply_button = tk.Button(self.root, text="Apply", command=self.apply_tolerance)
        self.apply_button.pack(pady=10)

        self.root.mainloop()

    def update_tolerance_display(self, event):
        self.tolerance_display.config(text=f"Tolerance: {self.tolerance_value.get()}")
        self.tolerance_entry.delete(0, tk.END)
        self.tolerance_entry.insert(0, str(self.tolerance_value.get()))

    def apply_tolerance(self):
        try:
            self.threshold = int(self.tolerance_entry.get())
        except ValueError:
            print("Invalid tolerance value. Using default.")
            self.threshold = self.tolerance_value.get()
        self.root.destroy()

    def onclick(self, event):
        if event.inaxes != self.ax:
            return
        seed_point = (int(event.xdata), int(event.ydata))
        mask = flood_fill(self.img.copy(), seed_point, self.threshold)
        contours = find_contours(mask)
        self.selected_contours.extend(contours)  # Add new contours to the list
        self.show_mask_with_contours()

    def onscroll(self, event):
        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()
        x_range = (x_max - x_min) / 2.0
        y_range = (y_max - y_min) / 2.0
        xdata, ydata = event.xdata, event.ydata

        if event.button == 'up':
            scale_factor = 1 / self.zoom_scale
        elif event.button == 'down':
            scale_factor = self.zoom_scale
        else:
            scale_factor = 1

        new_xrange = x_range * scale_factor
        new_yrange = y_range * scale_factor

        self.ax.set_xlim([xdata - new_xrange, xdata + new_xrange])
        self.ax.set_ylim([ydata - new_yrange, ydata + new_yrange])
        self.fig.canvas.draw()

    def show_mask_with_contours(self):
        combined_mask = np.zeros_like(self.img[:, :, 0], dtype=np.uint8)
        for contour in self.selected_contours:
            cv2.drawContours(combined_mask, [contour], -1, 255, thickness=cv2.FILLED)

        img_with_contours = self.original_img.copy()
        cv2.drawContours(img_with_contours, self.selected_contours, -1, (0, 255, 0), 2)
        self.ax.clear()
        self.ax.imshow(img_with_contours)
        self.fig.canvas.draw()

    '''def onhover(self, event):
        if event.inaxes != self.ax:
            return
        x, y = int(event.xdata), int(event.ydata)
        pixel_val = self.img[y, x]
        
        # Calculate the tolerance as the standard deviation of the neighborhood
        neighborhood = self.img[max(0, y-3):min(y+4, self.img.shape[0]), max(0, x-3):min(x+4, self.img.shape[1])]
        tolerance = np.std(neighborhood)
        
        # Round tolerance to the nearest integer
        tolerance = int(round(tolerance))
        
        print(f'Tolerance at ({x}, {y}): {tolerance}')'''

    def on_dest_click(self, event):
        if event.inaxes != self.dest_ax or not self.selected_contours:
            return
        dest_x, dest_y = int(event.xdata), int(event.ydata)
        self.paste_contours(dest_x, dest_y)

    def paste_contours(self, dest_x, dest_y):
        if not self.selected_contours:
            print("No contours to paste!")
            return

        for contour in self.selected_contours:
            x, y, w, h = cv2.boundingRect(contour)
            mask = np.zeros_like(self.img[:, :, 0], dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

            cut_contour = np.zeros((h, w, 4), dtype=np.uint8)
            cut_contour[:, :, :3] = self.original_img[y:y+h, x:x+w]
            cut_contour[:, :, 3] = mask[y:y+h, x:x+w]

            dest_img_with_alpha = np.zeros((self.destination_image.shape[0], self.destination_image.shape[1], 4), dtype=np.uint8)
            dest_img_with_alpha[:, :, :3] = self.destination_image
            dest_img_with_alpha[:, :, 3] = 255  # Set alpha channel to 255 (opaque)

            x1, y1 = dest_x, dest_y
            x2, y2 = x1 + w, y1 + h

            if x2 > self.destination_image.shape[1] or y2 > self.destination_image.shape[0]:
                raise ValueError("Contour exceeds destination image boundaries!")

            alpha_s = cut_contour[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            for c in range(0, 3):
                dest_img_with_alpha[y1:y2, x1:x2, c] = (alpha_s * cut_contour[:, :, c] +
                                                        alpha_l * dest_img_with_alpha[y1:y2, x1:x2, c])

            # Expand the region slightly for smoothing
            expanded_region_x1 = max(0, x1 - 5)
            expanded_region_y1 = max(0, y1 - 5)
            expanded_region_x2 = min(self.destination_image.shape[1], x2 + 5)
            expanded_region_y2 = min(self.destination_image.shape[0], y2 + 5)

            # Apply Gaussian blur to the expanded region
            dest_img_with_alpha[expanded_region_y1:expanded_region_y2, expanded_region_x1:expanded_region_x2, :3] = cv2.GaussianBlur(
                dest_img_with_alpha[expanded_region_y1:expanded_region_y2, expanded_region_x1:expanded_region_x2, :3], 
                (11, 11), 0
            )

            output_dest_path = os.path.join(self.output_dir, f'pasted_image.png')
            cv2.imwrite(output_dest_path, dest_img_with_alpha)
            print(f'Pasted contour onto destination image and saved to {output_dest_path}')

            self.destination_image = dest_img_with_alpha[:, :, :3]
            self.dest_ax.clear()
            self.dest_ax.imshow(dest_img_with_alpha)
            self.dest_fig.canvas.draw()

if __name__ == "__main__":
    image_path = r"C:\muhil\magicwand\1-RadNo_Y12345_W12345_FCAW_Thk14mm_SFD609mm_Exp30sec (2).png"
    destination_image_path = r"C:\muhil\output1\img5.png"
    tool = MagicWandTool(image_path, destination_image_path)
