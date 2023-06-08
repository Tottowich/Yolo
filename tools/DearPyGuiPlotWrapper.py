'''
DearPyGuiPlotWrapper.py
Import for easy plotting! Wraps all the boiler plate code. Look below for example usage.

Greger Burman, Adopticum, 2023.

Distributed under the Boost Software License, Version 1.0.
(See accompanying file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
'''

from sys import platform
from math import pi, sin
import dearpygui.dearpygui as dpg


class DearPyGuiPlotWrapper:
	_uuid_counter = 0

	# Think of config as static constants. Not changed at runtime.
	class _config:
		# Replace the ugly default pixel font with an acceptable font.
		# We load San Fransisco, which is the standard font in macOS.
		# We use a scale hack to get sharp text on retina/high DPI displays.
		font_for_macos = "/System/Library/Fonts/SFNS.ttf" # macOS
		font_for_winos = "C:/Windows/Fonts/verdana.ttf"   # Windows
		font_for_linux = "?"   # Linux
		font_size = 16
		font_scale = 2  # Oversample the font to get sharp text on retina displays.


	@classmethod
	def get_uuid(cls) -> int:
		cls._uuid_counter += 1
		return cls._uuid_counter


	# Boiler plate DearPyGui initialization.
	@classmethod
	def init(cls, window_title:str = "", window_width :int = 1280, window_height :int = 720):
		dpg.create_context()
		# Replace default ugly font with an acceptable font.
		font_file = cls._config.font_for_winos if (platform == "win32") else cls._config.font_for_macos if (platform == "darwin") else cls._config.font_for_linux
		with dpg.font_registry():
			dpg.set_global_font_scale(1.0/cls._config.font_scale)
			global_font = dpg.add_font(file=font_file, size=cls._config.font_scale * cls._config.font_size)
			dpg.bind_font(global_font)
		# Change some appearance with theme & style.
		with dpg.theme() as global_theme:
			with dpg.theme_component(dpg.mvAll):
				# Hack the window padding to allow image without borders.
				dpg.add_theme_style(dpg.mvStyleVar_WindowBorderSize, 0, category=dpg.mvThemeCat_Core)
				#dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 20, 20)
		dpg.bind_theme(global_theme)
		# Create the main viewport (what you elsewhere would call a "window").
		dpg.create_viewport(title=window_title, width=window_width, height=window_height, min_width=480, min_height=360)
		dpg.setup_dearpygui()
		dpg.show_viewport()


	@classmethod
	def run(cls, primary_window = None):
		# We're not using docking feature, we instead "dock" one window to the viewport.
		if primary_window:
			dpg.set_primary_window(primary_window, True)  #alt. use tag=guid

		# Setup either a managed render loop or an explicit render loop.
		dpg.start_dearpygui()  # Managed render loop.
		#while dpg.is_dearpygui_running(): #label="main_window"):  # Explicit render loop.

		# Cleanup
		dpg.destroy_context()


	@classmethod
	def example_static_plot(cls):
		cls.init(window_title="Plot of signal")

		# Define a dummy signal generator with sine overtones.
		def f(x=0, sine_coefficient_list=None):
			acc = 0
			for i, coeff in enumerate(sine_coefficient_list):
				acc += coeff * sin( (i+1) * x )
			return acc

		# Generate some sample data points.
		sample_count = 1000
		x_data = [i * 2*pi/sample_count for i in range(sample_count+1)]
		y_data = [f(x, [1.0, 0.0, 0.3, 0.0, 0.2, 0.0, 0.1]) for x in x_data]
		y_label = "f(x)"

		with dpg.window(no_scrollbar=True, no_close=True, no_collapse=True) as main_window:
			dpg.set_primary_window(main_window, True)
			with dpg.plot(height=-1, width=-1, anti_aliased=True, tracked=True):  #optionally: label="Diagram title"
				# Optionally create legend
				dpg.add_plot_legend(show=True, horizontal=True, location=dpg.mvPlot_Location_NorthWest)
				# REQUIRED: Create x and y axes.
				dpg.add_plot_axis(dpg.mvXAxis, label="x", tag="x_axis")
				with dpg.plot_axis(dpg.mvYAxis, label="y", tag="y_axis", log_scale=False):
					# Series belong to a y axis.
					dpg.add_line_series(x_data, y_data, label=y_label, parent="y_axis", tag="series1_tag")
					#dpg.add_scatter_series(x_data, y_data, label=y_label, parent="y_axis", tag="series1_tag")

		# Fully managed render loop.
		cls.run(primary_window=main_window)


	@classmethod
	def example_dynamic_plot(cls):
		cls.init(window_title="Plot of dynamic signal")

		# Define a dummy signal generator with sine overtones.
		def f(x, sine_coefficient_list=None):
			acc = 0
			for i, coeff in enumerate(sine_coefficient_list):
				acc += coeff * sin( (i+1) * x )
			return acc

		# Generate some sample data points.
		sample_count = 1000
		x_data = [i * 2*pi/sample_count for i in range(sample_count+1)]
		y_data = [f(x, []) for x in x_data]
		y_label = "f(x)"

		# Setup the UI element hierarchy.
		with dpg.window(no_scrollbar=True, no_close=True, no_collapse=True) as main_window:
			dpg.set_primary_window(main_window, True)  #alt. use tag=guid
			with dpg.plot(height=-1, width=-1, anti_aliased=True, tracked=True):  #optionally: label="Diagram title"
				# Optionally create legend
				dpg.add_plot_legend(show=True, horizontal=True, location=dpg.mvPlot_Location_NorthWest)
				# REQUIRED: Create x and y axes.
				dpg.add_plot_axis(dpg.mvXAxis, label="x", tag="x_axis")
				with dpg.plot_axis(dpg.mvYAxis, label="y", tag="y_axis", log_scale=False):
					dpg.set_axis_limits("y_axis", -1.0, 1.0)
					# Series belong to a y axis.
					dpg.add_line_series(x_data, y_data, label=y_label, parent="y_axis", tag="series1_tag")
					#dpg.add_scatter_series(x_data, y_data, label=y_label, parent="y_axis", tag="series1_tag")

		# Setup an explicit render loop, where we can run code between each frame.
		t = 0.0
		t_step = 0.03333
		while dpg.is_dearpygui_running():  # Explicit render loop.
			t = t + t_step  #if t < t_wrap else t + t_step - t_wrap

			# Calculate some coefficient weights for the signal generation.
			# on/off: sin(t) > 0
			# Tight rope:
			#coeffs = [0, sin(t*t) / (1 + t*t)]
			# Ocean
			#coeffs = [(0.488 * (1.05 + sin(0.5 * t))) * sin(0.5 * t)]
			# Snore
			#coeffs = [0]*21 + [(sin(t) > 0) * sin(t) * 0.1 * sin(15*t)]
			# Come and go:
			#coeffs = [0] * 20 + [0.1 * (sin(t) > 0) * sin(t)*sin(9*t)]
			# Ripple comes and goes
			coeffs = [0.7 * sin(t)] + [0]*20 + [(sin(0.5 * t) > 0) * sin(t) * 0.1 * sin(9*t)]

			# Update the data and plot, using the f(x) function on each sample.
			y_data = [f(x, coeffs) for x in x_data]
			dpg.set_value('series1_tag', [x_data, y_data])
			dpg.render_dearpygui_frame()

		# Cleanup
		dpg.destroy_context()


if "__main__" == __name__:
	DearPyGuiPlotWrapper.example_dynamic_plot()
