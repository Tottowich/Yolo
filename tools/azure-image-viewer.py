'''
azure-image-viewer.py
Query OCR-results from table in Azure.
Show the data as a table using DearPyGui.
When an item is selected, download the image from Azure blob storage and show it.

# Requirements:
# 	azure-data-tables
# 	azure-storage-blob
# 	dearpygui

Greger Burman, Adopticum, 2023.

Distributed under the Boost Software License, Version 1.0.
(See accompanying file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
'''

import sys
import os
#dep. import tempfile
import inspect
from datetime import date, timedelta
import dearpygui.dearpygui as dpg
from DearPyGuiPlotWrapper import DearPyGuiPlotWrapper as dpw
import cv2 as cv
import numpy as np

# Global runtime state.
class prop:
	datum = date.today()
	tmpdir = "."
	items = []
	selectables = []
	class image_object: # aka. the "viewer"
		id = None
		container_id = None
	class image_object: # aka. the "viewer"
		id = None
		container_id = None
	class texture:
		id = None
		width = 0
		height = 0
		ratio = 0
	class date_label:
		id = None
	class radio_non_zero:
		id = None
	class radio_adjust:
		id = None

def fit_image_in_container():
	if not prop.image.id:
		return

	# Get the curernt size of the image's parent item (container).
	width, height = dpg.get_item_rect_size(prop.image.container_id)
	view_ratio = width / height if height > 0 else 0
	print(f"Image container width={width} height={height} ratio={view_ratio:.2f}")

	# Calculate the size and position to fit the image.
	pad = 10
	x = 0
	y = 0
	if (prop.texture.ratio > view_ratio):
		# Do top+bottom margins.
		view_w = width - pad
		view_h = view_w * prop.texture.height // prop.texture.width
		y = (height - view_h) // 2
	else:
		# Do left+right margins.
		view_h = height - pad
		view_w = view_h * prop.texture.width // prop.texture.height
		x = (width - view_w) // 2
	# Set the position and size of the image.
	dpg.set_item_pos(prop.image.id, [x, y])
	dpg.set_item_width(prop.image.id, view_w)
	dpg.set_item_height(prop.image.id, view_h)


# This is currently pointless, because dark details in the small annotated images are already destroyed by jpeg compression. 
def load_image_with_cv(filepath:str):
	frame = cv.imread(filepath)
	(h, w, ch) = frame.shape[:3]
	# Convert pixels from BGR to RGB. Flatten matrix into a 1D array. 
	rgba = cv.cvtColor(frame, cv.COLOR_BGR2RGBA).ravel()
	# Convert values for GPU (float and normalized).
	rgba_float = np.true_divide(rgba, 255.0)
	mean_brightness = np.sum(rgba_float) / (w * h * ch)
	print(f"mean_brightness={mean_brightness}")

	# Attempt to auto adjust the "brightness" of the image.
	if dpg.get_value(prop.radio_adjust.id) == "Auto" and mean_brightness > 0:
		minimum_brightness = 0.66
		alpha = minimum_brightness / mean_brightness
		if alpha > 1:
			rgba_float = cv.convertScaleAbs(rgba_float, alpha = alpha, beta = 0)
			new_brightness = np.sum(rgba_float) / (w * h * ch)
			print(f"Auto adjust brightness. alpha={alpha}, new_brightness={new_brightness}")
	return (w, h, ch, rgba_float)


def load_image(filepath:str):
	# Load image directly with dpg: 
	result = dpg.load_image(filepath)
	#result = load_image_with_cv(filepath) #TODO:

	# Delete the old image and old texture.
	if prop.image.id:
		dpg.delete_item(prop.image.id)
		prop.image.id = None
	if prop.texture.id:
		dpg.delete_item(prop.texture.id)
		prop.texture.id = None
	if result is None:
		print("Failed to load image")
		return

	w, h, channels, texture_data = result
	print(f"Loaded image from file: {filepath}, {w}x{h}, {len(texture_data)//1024} kB")
	# Add texture to the texture registry. Optionally show a window with the registry.
	with dpg.texture_registry():
		prop.texture.id = dpg.add_static_texture(width=w, height=h, default_value=texture_data) #, parent=tex_reg_id)
		prop.texture.width = w
		prop.texture.height = h
		prop.texture.ratio = w / h if h > 0 else 0
	prop.image.id = dpg.add_image(texture_tag=prop.texture.id, parent=prop.image.container_id)
	fit_image_in_container()
	# TODO: Check put this example with using a drawlist:
	# https://github.com/hoffstadt/DearPyGui/discussions/1668


def on_generic_callback(sender, app_data, user_data):
	print(inspect.currentframe().f_code.co_name)
	#print(f"sender: {sender}, app_data: {app_data} user_data: {user_data}")
	print(f"  With sender ({sender}) as id: label={dpg.get_item_label(sender)}, value={dpg.get_value(sender)}")
	print(f"  With app_data ({app_data}) as id: label={dpg.get_item_label(app_data)}, value={dpg.get_value(app_data)}")
	width, height = dpg.get_item_rect_size(app_data)
	print(f"  With app_data ({app_data}) as id: width={width} height={height}")


def on_item_resized(sender, app_data, user_data):
	on_generic_callback(sender, app_data, user_data)
	fit_image_in_container()


'''
Obsolete: Select by each column in each row in table.
def on_item_clicked(sender, app_data, user_data):
	# Extract info from arguments.
	mouse_btn = app_data[0]
	idx_str, field = app_data[1].split(".")
	print(f"mouse button: {mouse_btn}, item: {idx_str}, field: {field}")
	idx = int(idx_str)

	# Select/highlight the item/row clicked in the table.
	select_row(idx)
'''


def on_key_pressed(sender, app_data, user_data):
	print(f"sender: {sender}, app_data: {app_data} user_data: {user_data}")
	if app_data == dpg.mvKey_Left:
		print("Left arrow key pressed")
		load_date(step=-1)
	elif app_data == dpg.mvKey_Right:
		print("Right arrow key pressed")
		load_date(step=+1)
	elif app_data == dpg.mvKey_Up:
		print("Up arrow key pressed")
		select_prev_row()
	elif app_data == dpg.mvKey_Down:
		print("Down arrow key pressed")
		select_next_row()


def add_debug_tools():
	with dpg.menu_bar():
		with dpg.menu(label="Tools"):
			dpg.add_menu_item(label="Show About", callback=lambda:dpg.show_tool(dpg.mvTool_About))
			dpg.add_menu_item(label="Show Metrics", callback=lambda:dpg.show_tool(dpg.mvTool_Metrics))
			dpg.add_menu_item(label="Show Documentation", callback=lambda:dpg.show_tool(dpg.mvTool_Doc))
			dpg.add_menu_item(label="Show Debug", callback=lambda:dpg.show_tool(dpg.mvTool_Debug))
			#dpg.add_menu_item(label="Show Style Editor", callback=lambda:dpg.show_tool(dpg.mvTool_Style))
			#dpg.add_menu_item(label="Show Font Manager", callback=lambda:dpg.show_tool(dpg.mvTool_Font))
			dpg.add_menu_item(label="Show Item Registry", callback=lambda:dpg.show_tool(dpg.mvTool_ItemRegistry)) #dpg.show_item_registry()
			#TODO: How to open the my texture registry after the fact?

def setup_gui():
	# Setup the UI element hierarchy.
	with dpg.window(label="Main window", no_scrollbar=True, no_close=True, no_collapse=True) as main_window:
		dpg.set_primary_window(main_window, True)  #alt. use tag=guid

		# Setup theme for the items in the table.
		with dpg.theme() as selectable_theme:
			with dpg.theme_component(dpg.mvTable):
				dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 1, 5)
			with dpg.theme_component(dpg.mvSelectable):
				# Color when hovered by mouse cursor:
				#dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (191, 191, 255, 16), category=dpg.mvThemeCat_Core)
				# Color when mouse button is down:
				dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (191, 191, 255, 127), category=dpg.mvThemeCat_Core)
				# Color of selected item:
				dpg.add_theme_color(dpg.mvThemeCol_Header, (191, 191, 255, 63), category=dpg.mvThemeCat_Core)
		dpg.bind_theme(selectable_theme)

		# Hookup callback for when window is resized.
		with dpg.item_handler_registry(label="Main window handler registry") as main_window_handler_registry:
			dpg.add_item_resize_handler(label="Main window resize handler", callback=on_item_resized)
			dpg.bind_item_handler_registry(main_window, main_window_handler_registry)

		# Hookup callbacks to handle keypresses e.g. for navigating the table.
		with dpg.handler_registry(show=True):
			dpg.add_key_press_handler(key=dpg.mvKey_Up, callback=on_key_pressed)
			dpg.add_key_press_handler(key=dpg.mvKey_Down, callback=on_key_pressed)
			dpg.add_key_press_handler(key=dpg.mvKey_Left, callback=on_key_pressed)
			dpg.add_key_press_handler(key=dpg.mvKey_Right, callback=on_key_pressed)

		with dpg.child_window(label="Toolbar", width=-10, height=30, no_scrollbar=True, border=False):
			with dpg.group(horizontal=True):
				dpg.add_text("PartitionKey:")
				prop.date_label.id = dpg.add_text(prop.datum.isoformat())
				dpg.add_button(label="Yesterday", arrow=True, direction=dpg.mvDir_Left, callback=lambda: load_date(step=-1))
				dpg.add_button(label="Tomorrow", arrow=True, direction=dpg.mvDir_Right, callback=lambda: load_date(step=+1))
				dpg.add_spacer()
				prop.radio_non_zero.id = dpg.add_radio_button(["Non Zeros", "Zeros"], horizontal=True, callback=lambda: load_date(step=0))
				dpg.add_spacer()
				dpg.add_text("Select item")
				dpg.add_button(label="Button", arrow=True, direction=dpg.mvDir_Up, callback=select_row_new(delta=-1))
				dpg.add_button(label="Button", arrow=True, direction=dpg.mvDir_Down, callback=select_row_new(delta=+1))
				dpg.add_spacer()
				prop.radio_adjust.id = dpg.add_radio_button(["Original", "Auto"], horizontal=True, callback=select_row_new())

		# Setup a verticlly split layout which is resizable (impl by table + group + child_window). 
		with dpg.child_window(label="Main container", width=-1, height=-10, no_scrollbar=True, border=False):
			with dpg.table(label="Table layout", header_row=False, resizable=True):
				#borders_innerH=True, borders_outerH=True, borders_innerV=True, borders_outerV=True):
				dpg.add_table_column()
				dpg.add_table_column()
				with dpg.table_row(label="Table row 1"):
					# Column 1
					with dpg.group(label="Group 1") as group2:
						# Use child_window to fill the space, then add the content inside.
						with dpg.child_window(label="Container 1", width=-1, height=-1, no_scrollbar=True, border=True) as image_container:
							prop.image.container_id = image_container
						# Hookup callback for when table cell (group) is resized.
						with dpg.item_handler_registry(label="Group 1 handler registry") as group2_handler_registry:
							dpg.add_item_resize_handler(label="Group 1 resize handler", callback=on_item_resized)
							dpg.bind_item_handler_registry(group2, group2_handler_registry)
					# Column 2
					with dpg.group(label="Group 2") as group2:
						# Use child_window to fill the space, then add the content inside.
						with dpg.child_window(label="Container 2", width=-1, height=-1, no_scrollbar=True, border=True) as image_container:
							prop.image.container_id = image_container
						# Hookup callback for when table cell (group) is resized.
						with dpg.item_handler_registry(label="Group 2 handler registry") as group2_handler_registry:
							dpg.add_item_resize_handler(label="Group 2 resize handler", callback=on_item_resized)
							dpg.bind_item_handler_registry(group2, group2_handler_registry)


def main():
	tmpdir = "/tmp/azure-image-viewer"
	print('Temporary directory: ', tmpdir)
	prop.tmpdir = tmpdir

	if len(sys.argv) > 1:
		try: # Parse an ISO date from the 1st argument.
			prop.datum = date.fromisoformat(sys.argv[1])
		except Exception as ex:
			print("1st argument is not a valid date in YYYY-MM-DD format.")
			print(f"Exception: {ex}")

	dpw.init(window_title="Image Viewer")
	setup_gui()
	load_date(datum=prop.datum)
	dpw.run()


if "__main__" == __name__:
	main()
