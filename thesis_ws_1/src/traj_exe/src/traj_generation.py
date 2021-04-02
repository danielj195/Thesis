#!/usr/bin/env python
import numpy as np 
import csv
import rospy
import math
import copy
import logging
import matplotlib
import matplotlib.lines as mlines
import threading
import sys
import random

# import plotly.figure_factory as ff
# from scipy.spatial import Delaunay

from shapely.geometry import Polygon
from matplotlib import cm
from matplotlib import pyplot as plt

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import *
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from mpl_toolkits.mplot3d import Axes3D
from numpy import sin, cos, pi

from geomdl import fitting
from geomdl import NURBS
from geomdl import BSpline
from geomdl import utilities
from geomdl import evaluators
from geomdl import operations
from geomdl import exchange 
from geomdl import knotvector
from geomdl.visualization import VisMPL as vis
from geomdl import construct, multi

from sklearn.linear_model import LinearRegression



import numpy
from stl import mesh

np.set_printoptions(threshold=sys.maxsize)


class Traj_Gen:


	def __init__(self):
		self.hatch_angle = 1.57
		self.wp_spacing = 0.254 #meters
		self.hatch_spacing = 0.05
		self.radius = 0.0381 #0.064 #2  #radius of sanding pad inches (in meters)
		self.ecc = 0.000027 #0.000127 #0.005  #eccentricity in inches
		self.overlap = 0.0
		self.direction = 'u'  #Direction of initial seed curve
		self.increasing = True #False  #True if parametric direction inceases with euclidean direction; False otherwise

		self.equal_spacing = 0.02 #0.05 #0.02 #0.0125 #0.0508 #0.0125 #0.00757
		self.first_last_offset = 0.10
		self.force_pt_offset = 0.005

		# self.splines = []
		self.seed_pts = []
		self.surf_width = 0.0
		self.surf_length = 0.0
		self.seed_uv_pts = []
		self.seed_arr = []
		self.adj_pts = []
		self.edge_pts = []   #Used for determining placement of next ellipse


		#Trajectory Lists
		self.uv_pts = []
		self.cart_pts = []
		self.cross_uv_pts = []
		self.cross_cart_pts = []
		

		self.uv_pts_row = []
		self.max_y_list = []
		self.min_y_list = []
		self.new_max_y_list = []
		self.new_min_y_list = []
		self.prev_max_y_list = []
		self.prev_min_y_list = []
		# self.k_1_dir = []
		# self.k_2_dir = []

		self.covered = False
		# self.iter_max = 50

		#Trajectory Size
		self.num_rows = 30 #10
		self.num_rows_equal = 4
		self.seed_min = 0.15
		self.seed_max = 0.85
		self.u_start = np.array([0.07]) 

		#Trajectory Length
		self.traj_total_length = 0.0
 


		#Initialize Surface
		self.surf_init = BSpline.Surface()
		self.surf_init.degree_u = 3
		self.surf_init.degree_v = 3

		#Initialize Smooth Surface
		self.surf_smooth = BSpline.Surface()
		self.surf_smooth.degree_u = 3
		self.surf_smooth.degree_v = 3


		self.u_num = 48  #rows in x direction
		self.v_num =15   #columns in y direction

		#Generate Real Surface
		# self.data = self.import_data()
		# self.surf_init.set_ctrlpts(self.data, self.u_num, self.v_num)
		# self.surf_init.knotvector_u = utilities.generate_knot_vector(self.surf_init.degree_u, self.u_num)
		# self.surf_init.knotvector_v = utilities.generate_knot_vector(self.surf_init.degree_v, self.v_num)
		# self.surf_init.delta = 0.005
		# self.surf_init.evaluate()
		# self.points = self.surf_init.evalpts
		# print(self.points)

		#Least Squares Surface
		# self.surf_init = fitting.approximate_surface(self.points, self.u_num, self.v_num, 6, 6) 
		# self.surf_init.delta = 0.025

		#Smooth Surface
		# self.smooth_data = self.smooth_surface()
		# self.surf_smooth.set_ctrlpts(self.smooth_data, self.u_num-1, self.v_num)
		# self.surf_smooth.knotvector_u = utilities.generate_knot_vector(self.surf_smooth.degree_u, self.u_num-1)
		# self.surf_smooth.knotvector_v = utilities.generate_knot_vector(self.surf_smooth.degree_v, self.v_num)
		# self.surf_smooth.delta = 0.025



		# self.points = self.surf_init.evaluate(start_u=0.01,stop_u=0.99,start_v=0.01,stop_v=0.99)
		

		#Generate Fake Surface
		self.surf = BSpline.Surface()
		self.surf.degree_u = 3
		self.surf.degree_v = 3
		self.surf.set_ctrlpts(*exchange.import_txt(r"/home/daniel/thesis_ws/src/traj_gen/scripts/ex_surface01.cpt", two_dimensional=True))
		self.surf.knotvector_u = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0]
		self.surf.knotvector_v = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0]
		self.surf.delta = 0.01
		self.surf.evaluate()
		self.points = self.surf.evalpts
		# print("Surface Points: ", self.points)


		#Rotated Surface
		# self.surf = operations.rotate(self.surf_smooth, 90)
		# self.surf.delta = 0.025


		#Raw Surface (before regression)
		self.fig0 = plt.figure(figsize=plt.figaspect(0.2)*1.5) 
		self.ax0 = Axes3D(self.fig0)

		#Figure with Ellipse
		self.fig = plt.figure(figsize=plt.figaspect(0.2)*1.5) 
		# self.ax = self.fig.gca(projection='3d')
		self.ax = Axes3D(self.fig)
		self.ax.set_xlim([-0.4,0.4])
		self.ax.set_ylim([-0.5,0.2])
		self.ax.set_zlim([-0.35,0.0])

		#Figure without Ellipse
		self.fig1 = plt.figure(figsize=plt.figaspect(0.2)*1.5) 
		self.ax1 = Axes3D(self.fig1)

		#Smooth Surface
		self.fig2 = plt.figure(figsize=plt.figaspect(0.2)*1.5) 
		self.ax2 = Axes3D(self.fig2)


		self.threads = list()
		self.threads_equal = list()

		#Spline variables
		# self.scan_splines = multi.CurveContainer()
		self.cross_way_pts = []
		self.scan_way_pts = []

		self.scan_normals = []
		self.cross_normals = []

		self.plot_ellipse = False
		self.plot_area = True
		self.plot_edge = False

		self.cover_area_tot = 0.0
		self.overlap_area_tot = 0.0
		self.uncovered_area_tot = 0.0

		plt.rcParams['font.size'] = '30'


		# self.dist = 0.05 #distance between way points

	def import_csv(self,filename):
		with open(filename) as data_file:
			data = list(csv.reader(data_file,delimiter=','))
		return data


	# Description:   This function is used for smoothing the 3D surface using linear regression.  Not needed for perfectly smooth surfaces (CAD)
	def smooth_surface(self):
		space_pts_list = self.import_csv(r'/home/daniel/thesis_ws/src/traj_exe/src/data/spaced_data.csv')
		space_pts_arr = np.asarray(space_pts_list)[2:,:].astype(np.float)

		# space_pts_arr = np.linspace(0.7372, 1.3427, self.u_num)
		pts_arr = np.asarray(self.points)
		print(pts_arr)
		surf_pts = []

		#Get x direction boundaries

		dim_min = self.surf_init.evaluate_single([0,0])
		dim_max = self.surf_init.evaluate_single([1.0, 1.0])
		y_min = dim_min[1]
		y_max = dim_max[1]
		y_pred = np.linspace(y_min, y_max, self.v_num)[:, np.newaxis]
		#Perform Linear Regression

		for i in range(space_pts_arr.shape[0]-1):
			lb = space_pts_arr[i] - 2e-3
			ub = space_pts_arr[i] + 2e-3
			print("lb: ", lb)
			print("ub: ", ub)
			row = pts_arr[(pts_arr[:,0] > lb) & (pts_arr[:,0] < ub),:]
			X = row[:,1][:, np.newaxis]
			y = row[:,2][:, np.newaxis]
			print("X shape: ", X.shape)
			print("y shape: ", y.shape)
			
			x_pred = np.full(self.v_num, space_pts_arr[i])[:, np.newaxis]
			# print(x_pred.shape)
			# print(y_pred.shape)
			reg = LinearRegression().fit(X,y)

			z_pred = reg.predict(y_pred)
			
			pred_pts = np.hstack((x_pred,y_pred,z_pred))
			pred_pts_list = pred_pts.tolist()
			surf_pts.extend(pred_pts_list)
			# print("pred_pts: ", pred_pts)

		points = np.asarray(surf_pts)
		print("points: ", points)
		
		points = points[points[:,0].argsort()] # First sort doesn't need to be stable.
		# print("pre sorted: ", points)



		sorted_points = np.empty(points.shape)
		# y_coords = np.empty((self.v_num, self.u_num))
		for i in range(self.u_num-1):				#sort points
			point_row = points[i*self.v_num:i*self.v_num+self.v_num,:]
			point_row = point_row[point_row[:,1].argsort()]
			sorted_points[i*self.v_num:i*self.v_num+self.v_num,:] = point_row

		points_list = sorted_points.tolist()
		return points_list



	#Description:  This function is used to import data collected using a point laser.  (Not needed for CAD)
	def import_data(self):
		data = self.import_csv(r'/home/daniel/thesis_ws/src/traj_exe/src/data/surface_data.csv')
		points = np.asarray(data)[30:,:].astype(np.float)
		# R = np.array([[0.0, -1.0, 0.0],
		# 			  [1.0, 0.0, 0.0],
		# 			  [0.0, 0.0, 1.0]])
		# for i in range(points.shape[0]):    	#Rotate points by 90 degrees for better NURBS accuracy
		# 	points[i,:] = R.dot(points[i,:])

		filter_data = []

		points = points[points[:,0].argsort()] # First sort doesn't need to be stable.
		# print("pre sorted: ", points)



		sorted_points = np.empty(points.shape)
		y_coords = np.empty((self.v_num, self.u_num))
		for i in range(self.u_num):				#sort points
			point_row = points[i*self.v_num:i*self.v_num+self.v_num,:]
			point_row = point_row[point_row[:,1].argsort()]

			#Multiple Linear Regression
			# X = point_row[:,:2]
			# y = point_row[:,2]
			# print("X: ", X)
			# print("y: ", y)
			# reg = LinearRegression().fit(X,y)
			# pred = reg.predict(point_row[:,:2])
			# print("pred: ", pred)
			# point_row[:,2] = pred


			# print("point row: ", point_row)
			'''
			point_x_mean = np.mean(point_row[:,0])    #Find mean x and replace x-coordinates with mean
			point_row[:,0] = point_x_mean*np.ones(self.v_num)
			'''
			# # print("point row: ", point_row)
			# point_z_mean = np.mean(point_row[:,2])    #Find mean z and replace z-coordinates with mean
			# # point_row[:,2] = point_z_mean*np.ones(self.v_num)
			# # print("point row: ", point_row)
			# filter_data.append([point_x_mean, point_z_mean])

			sorted_points[i*self.v_num:i*self.v_num+self.v_num,:] = point_row
			'''
			y_coords[:,i] = point_row[:,1]
			'''

		# filter_data_arr = np.asarray(filter_data)
		# print("Filter Data: ", filter_data_arr)
		# # fig2 = plt.figure(figsize=plt.figaspect(0.2)*1.5)
		# ax2 = plt.subplot()
		# ax2.plot(filter_data_arr[:,0], filter_data_arr[:,1])

		# w = scipy.fftpack.rfft(filter_data_arr[:,1])
		# f = scipy.fftpack.rfftfreq(filter_data_arr.shape[0], filter_data_arr[1,0]-filter_data_arr[0,0])
		# spectrum = w**2
		# print("spectrum: ", spectrum)

		# ax4 = plt.subplot()
		# ax4.plot(filter_data_arr[:,0], spectrum)

		# cutoff_idx = spectrum < (spectrum.max()/8)
		# w2 = w.copy()
		# w2[cutoff_idx] = 0

		# y2 = scipy.fftpack.irfft(w2)

		# ax3 = plt.subplot()
		# ax3.plot(filter_data_arr[:,0], y2)

		# print("y2: ", y2)

		'''
		avg_y_coord = np.mean(y_coords, axis=1)
		# # print("avg_y_coord: ", avg_y_coord)

		for j in range(self.u_num):
			sorted_points[j*self.v_num:j*self.v_num+self.v_num,1] = avg_y_coord


		for j in range(self.u_num):
			point_row = points[j*self.v_num:j*self.v_num+self.v_num,:]
			point_row = point_row[point_row[:,1].argsort()]

			#Multiple Linear Regression
			X = point_row[:,:2]
			y = point_row[:,2]
			# print("X: ", X)
			# print("y: ", y)
			reg = LinearRegression().fit(X,y)
			pred = reg.predict(sorted_points[j*self.v_num:j*self.v_num+self.v_num,:2])
			# print("pred: ", pred)
			sorted_points[j*self.v_num:j*self.v_num+self.v_num,2] = pred
		'''


		# print("Sorted Points: ", sorted_points)
		points_list = sorted_points.tolist()
		


		# points_list = points.tolist()



		# ax4 = plt.subplot()
		# ax4.scatter(sorted_points[:,0], sorted_points[:,2])


		return points_list


	#Description:  Finds the dimensions of the surface
	def surf_dim(self):
		self.dim_min = self.surf.evaluate_single([0,0])
		self.dim_max = self.surf.evaluate_single([1.0, 1.0])
		print("surf min: ", self.dim_min)
		print("surf max: ", self.dim_max)
		self.surf_width = abs(self.dim_max[1] - self.dim_min[1])
		self.surf_length = abs(self.dim_max[0] - self.dim_min[0])
		# self.spline_pt_spacing = 0.05*self.surf_width
		self.spline_pt_spacing = 0.03*self.surf_width


	#Description:  Finds the seed (initial) curve on the surface. Either in u or v direction
	def seed_curve(self, uv_val):
		del self.uv_pts[:]
		del self.seed_pts[:]
		seed_uv_pts_row = []
		seed_pts_row = []


		if self.direction == 'u':
			u = self.seed_min
			# v = 0.5
			# du = 0.05*self.spline_pt_spacing/self.surf_length
			du = 0.01*self.spline_pt_spacing/self.surf_length

			# du_incr = du/50.0
			#dv = 0.05*self.hatch_spacing/self.surf_width
			out_of_bounds = False
			prev_pt = self.surf.evaluate_single([u, uv_val])
			seed_pts_row.append(prev_pt)
			seed_uv_pts_row.append([u, uv_val])
			while not out_of_bounds:
				dist = 0.0 #current distance between points
				while dist < self.spline_pt_spacing:
					next_pt = self.surf.evaluate_single([u, uv_val])
					next_pt_arr = np.asarray(next_pt)
					prev_pt_arr = np.asarray(prev_pt)
					incr_dist = np.linalg.norm(next_pt_arr - prev_pt_arr)
					dist = dist + incr_dist
					prev_pt = next_pt
					u = u + du
					if u > self.seed_max:
						out_of_bounds = True
						break
				if not out_of_bounds:
					seed_uv_pts_row.append([u, uv_val])
				seed_pts_row.append(prev_pt)

		if self.direction == 'v':
			v = self.seed_min
			# v = 0.5
			# dv = 0.05*self.spline_pt_spacing/self.surf_length
			dv = 0.01*self.spline_pt_spacing/self.surf_length
			print("dv: ", dv)

			# du_incr = du/50.0
			#dv = 0.05*self.hatch_spacing/self.surf_width
			out_of_bounds = False
			prev_pt = self.surf.evaluate_single([uv_val, v])
			seed_pts_row.append(prev_pt)
			seed_uv_pts_row.append([uv_val, v])
			while not out_of_bounds:
				dist = 0.0 #current distance between points
				while dist < self.spline_pt_spacing:
					next_pt = self.surf.evaluate_single([uv_val, v])
					next_pt_arr = np.asarray(next_pt)
					prev_pt_arr = np.asarray(prev_pt)
					incr_dist = np.linalg.norm(next_pt_arr - prev_pt_arr)
					dist = dist + incr_dist
					prev_pt = next_pt
					v = v + dv
					if v > self.seed_max:
						out_of_bounds = True
						break
				if not out_of_bounds:
					seed_uv_pts_row.append([uv_val, v])
				seed_pts_row.append(prev_pt)

		self.seed_pts.append(seed_pts_row)
		self.uv_pts.append(seed_uv_pts_row)
		self.seed_uv_pts = seed_uv_pts_row
		# print("seed uv pts", self.seed_uv_pts)

		self.seed_arr = np.asarray(self.seed_pts[0])
		# print(self.seed_arr)


	#Description:  Plots the surface
	def plot_surface(self):
		self.surf.evaluate(start_u=0.005, start_v=0.005)
		self.surface_points = self.surf.evalpts
		self.data = np.asarray(self.surface_points)
		# fig = plt.figure(figsize=plt.figaspect(0.2)*1.5) 
		# ax = Axes3D(fig)
		my_cmap = plt.get_cmap('hot')
		# self.ax.plot_trisurf(self.data[:,0],self.data[:,1],self.data[:,2],cmap = my_cmap, alpha=0.5)
		# self.ax1.plot_trisurf(self.data[:,0],self.data[:,1],self.data[:,2],cmap = my_cmap, alpha=0.5)
		# self.ax.plot_trisurf(self.data[:,0],self.data[:,1],self.data[:,2], alpha=0.5, color='y',edgecolor='y', zorder=0)
		# self.ax1.plot_trisurf(self.data[:,0],self.data[:,1],self.data[:,2], alpha=0.5, color='y',edgecolor='y', zorder=0)
		self.ax.plot_trisurf(self.data[:,0],self.data[:,1],self.data[:,2], alpha=1.0, color='y',linewidth=0, edgecolor='y', zorder=0)
		self.ax1.plot_trisurf(self.data[:,0],self.data[:,1],self.data[:,2], alpha=1.0, color='y',linewidth=0, edgecolor='y', zorder=0)
		# self.ax.plot_surface(self.data[:,0],self.data[:,1],self.data[:,2], alpha=1.0, color='y')
		# self.ax1.plot_surface(self.data[:,0],self.data[:,1],self.data[:,2], alpha=1.0, color='y')

		# X = np.arange(-5, 5, 0.01)
		# Y = np.arange(-5, 5, 0.01)
		# X, Y = np.meshgrid(X, Y)
		# R = np.sqrt(X**2 + Y**2)
		# Z = np.sin(R)

		# # Plot the surface.
		# surf = self.ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
		#                        linewidth=0, antialiased=False)


		'''

		self.surf_init.delta = 0.025
		self.surface_init_points = self.surf_init.evalpts
		self.data_init = np.asarray(self.surface_init_points)
		self.ax0.plot_trisurf(self.data_init[:,0],self.data_init[:,1],self.data_init[:,2],cmap = my_cmap, alpha=0.5)
		'''

		'''
		self.surf_smooth.evaluate()
		self.smooth_surface_points = self.surf_smooth.evalpts
		self.smooth_data = np.asarray(self.smooth_surface_points)
		# fig = plt.figure(figsize=plt.figaspect(0.2)*1.5) 
		# ax = Axes3D(fig)
		# my_cmap = plt.get_cmap('hot')
		self.ax2.plot_trisurf(self.smooth_data[:,0],self.smooth_data[:,1],self.smooth_data[:,2],cmap = my_cmap, alpha=0.5)
		# self.ax1.plot_trisurf(self.data[:,0],self.data[:,1],self.data[:,2],cmap = my_cmap, alpha=0.5)
	

		self.surf.evaluate()
		self.rot_surface_points = self.surf.evalpts
		self.rot_data = np.asarray(self.rot_surface_points)
		# fig = plt.figure(figsize=plt.figaspect(0.2)*1.5) 
		# ax = Axes3D(fig)
		# my_cmap = plt.get_cmap('hot')
		self.ax2.plot_trisurf(self.rot_data[:,0],self.rot_data[:,1],self.rot_data[:,2],cmap = my_cmap, alpha=0.5)
		'''





		# # uv_vals = [[0.1, 0.1], [0.1, 0.3], [0.1, 0.5], [0.1, 0.7]]
		# uv_vals = [[0.1, 0.1], [0.2, 0.1], [0.7, 0.1]]
		# surfnorms = [[] for _ in range(len(uv_vals))]

		# for idx, uv in enumerate(uv_vals):
		# 	surfnorms[idx] = operations.normal(self.surf, uv, normalize=True)

		# normal_vectors = np.array(surfnorms)


		# self.ax.quiver(normal_vectors[:, 0, 0], normal_vectors[:, 0, 1], normal_vectors[:, 0, 2],
		# normal_vectors[:, 1, 0], normal_vectors[:, 1, 1], normal_vectors[:, 1, 2],
		# color='red', length=3)


	#Description:  Thread function for finding adjacent way point for uniform spacing
	def thread_func(self, i, w, v, uv_pt, row_num, boundary):
		adj_pt, new_max_y, new_min_y = self.adjacent_pt(uv_pt, boundary, i)
		if new_max_y[1] >= 20:
			self.covered = True
			# print("FINISHED")
		if np.isnan(new_max_y).any():
			print("point OOB")
			# self.uv_pts[row_num].pop(i)
		else:
			self.uv_pts_row.append(adj_pt)
			self.new_max_y_list.append(new_max_y)
			self.new_min_y_list.append(new_min_y)


	#Description:  Thread function for finding adjacent way point for equal spacing		
	def thread_func_equal(self, uv_pt):
		adj_pt, new_max_y, new_min_y = self.adjacent_pt_equal(uv_pt)
		if new_max_y[1] >= 20:
			self.covered = True
			# print("FINISHED")
		if np.isnan(new_max_y).any():
			print("point OOB")
			# self.uv_pts[row_num].pop(i)
		else:
			self.uv_pts_row.append(adj_pt)
			self.new_max_y_list.append(new_max_y)
			self.new_min_y_list.append(new_min_y)
		# adj_pt = self.adjacent_pt_equal(uv_pt)
		# self.uv_pts_row.append(adj_pt)
	

	#Description:   Finds the base points (ellipse centers) for uniform spacing
	def base_pts(self):
		# del self.k_1_dir[:]
		# del self.k_2_dir[:]
		del self.adj_pts[:]
		row_num = 0
		# uv_pts = self.uv_pts[row_num]
		uv_pts = copy.deepcopy(self.uv_pts[row_num])
		self.max_y_list = [0,0,0]*len(uv_pts)




		#Boundary Spline for very first spline
		# first_max_y_list = []
		for i in range(len(uv_pts)):
			w, v = self.shape_operator(uv_pts[i])
			print("w: ", w)
			print("v: ", v)
			max_y, min_y = self.ribbon_width(w, v, uv_pts[i], self.plot_ellipse)   #max_y is array
			print("max_y", max_y)
			self.prev_max_y_list.append(max_y.tolist())
			self.prev_min_y_list.append(min_y.tolist())

		# self.prev_max_y_list.pop(0)
		# self.prev_max_y_list.pop(-1)
		print("prev_max_y_list", self.prev_max_y_list)
		edge_curve = BSpline.Curve() 
		edge_curve.degree = 5
		edge_curve.ctrlpts = copy.deepcopy(self.prev_max_y_list)
		edge_curve.knotvector = utilities.generate_knot_vector(edge_curve.degree, len(edge_curve.ctrlpts))
		edge_curve.delta = 0.001
		edge_curve.evaluate()
		edge_curve_pts = edge_curve.evalpts
		self.edge_pts.append(edge_curve_pts)
		# print("edge_curve_pts", edge_curve_pts)

		edge_curve_arr = np.asarray(edge_curve_pts)

		if self.plot_edge == True: 
			self.ax.scatter(edge_curve_arr[:,0], edge_curve_arr[:,1], edge_curve_arr[:,2], color='blue')



		#Continue with remaining splines
		row_num = 1


		# while not self.covered:
		for k in range(self.num_rows):
		# while (self.cover_area_tot + self.uncovered_area_tot - self.overlap_area_tot) < 0.745:
			print("AREA: ", self.cover_area_tot - self.overlap_area_tot)
			print("TRAJECTORY ROW: ", row_num)
			for i in range(len(uv_pts)):
				w, v = self.shape_operator(uv_pts[i])
				x = threading.Thread(target=self.thread_func, args=(i,w,v,uv_pts[i],row_num,self.edge_pts[row_num-1]))
				self.threads.append(x)
				x.start()

			for index, thread in enumerate(self.threads):
				logging.info("Main    : before joining thread %d.", index)
				thread.join()
				logging.info("Main    : thread %d done", index)



			pts_array = np.asarray(self.uv_pts_row)
			new_max_y_array = np.asarray(self.new_max_y_list)
			new_min_y_array = np.asarray(self.new_min_y_list)
			concat_array = np.concatenate((pts_array, new_max_y_array, new_min_y_array), axis=1)
			if self.direction == 'u':
				concat_array = concat_array[np.argsort(concat_array[:, 0])]
			if self.direction == 'v':
				concat_array = concat_array[np.argsort(concat_array[:, 1])]
			pts_array = concat_array[:,0:2]
			new_max_y_array = concat_array[:,2:5]
			new_min_y_array = concat_array[:,5:8]

			prev_max_y_array = np.asarray(self.prev_max_y_list)
			prev_min_y_array = np.asarray(self.prev_min_y_list)

			prev_max_y_array = prev_max_y_array[prev_max_y_array[:,0].argsort()]
			prev_min_y_array = prev_min_y_array[prev_min_y_array[:,0].argsort()]

			# Find Coverage Area

			# for i in range(prev_max_y_array.shape[0]-1):
			for i in range(new_min_y_array.shape[0]-1):
				pt0 = prev_min_y_array[i,:]
				pt1 = prev_max_y_array[i,:]
				pt2 = prev_max_y_array[i+1,:]
				pt3 = prev_min_y_array[i+1,:]
				#New minimum points
				pt4 = new_min_y_array[i,:]
				pt5 = new_min_y_array[i+1,:]
				#Covered area
				cover_pts = np.vstack((pt0, pt1, pt2, pt3))
				# cover_pts_list = cover_pts.tolist()
				# polygon_1 = Polygon(cover_pts_list)
				# cover_area = polygon_1.area
				cover_area = self.area_slice(cover_pts, self.plot_area, 'b')
				self.cover_area_tot = self.cover_area_tot + cover_area
				#Overlap area
				overlap_pts = np.vstack((pt1, pt2, pt5, pt4))
				# overlap_pts_list = overlap_pts.tolist()
				# polygon_2 = Polygon(overlap_pts_list)
				# overlap_area = polygon_2.area
				overlap_area = self.area_slice(overlap_pts, self.plot_area, 'r')
				self.overlap_area_tot = self.overlap_area_tot + overlap_area

			pts_list = pts_array.tolist()
			# print("pts_list", pts_list)


			max_y_list = new_max_y_array.tolist()


			if self.increasing == False:
				max_y_list = max_y_list[::-1]

			self.uv_pts.append(pts_list)
			# print("uv_pts", self.uv_pts)

			# del self.max_y_list[:]
			del uv_pts[:]

			del self.prev_max_y_list[:]
			del self.prev_min_y_list[:]
			self.prev_max_y_list = []
			self.prev_min_y_list = []
			self.prev_max_y_list = copy.deepcopy(self.new_max_y_list)
			self.prev_min_y_list = copy.deepcopy(self.new_min_y_list)
			del self.new_max_y_list[:]
			del self.new_min_y_list[:]
			self.new_max_y_list = []
			self.new_min_y_list = []

			del self.uv_pts_row[:]


			# self.max_y_list = []
			uv_pts = []
			# self.new_max_y_list = []
			self.uv_pts_row = []
			# self.edge_pts = []


			# for i in range(len(max_y_list)):
			# 	self.max_y_list.append(max_y_list[i])

			for i in range(len(pts_list)):
				uv_pts.append(pts_list[i])



			#Form ribbon edge boundary
			# print("max_y_list", max_y_list)
			edge_curve = BSpline.Curve() 
			edge_curve.degree = 5
			edge_curve.ctrlpts = copy.deepcopy(max_y_list)
			edge_curve.knotvector = utilities.generate_knot_vector(edge_curve.degree, len(edge_curve.ctrlpts))
			edge_curve.delta = 0.001
			edge_curve.evaluate()
			edge_curve_pts = edge_curve.evalpts
			self.edge_pts.append(edge_curve_pts)
			# print("edge_curve_pts", edge_curve_pts)

			edge_curve_arr = np.asarray(edge_curve_pts)

			if self.plot_edge == True: 
				self.ax.scatter(edge_curve_arr[:,0], edge_curve_arr[:,1], edge_curve_arr[:,2], color='yellow')

			row_num = row_num + 1


			print("concat_array", concat_array)
			print("pts_array", pts_array)
			print("max_y_array", new_max_y_array)

	#Description:   Finds the base points (ellipse centers) for equal spacing
	def base_pts_equal(self):
		# del self.k_1_dir[:]
		# del self.k_2_dir[:]
		del self.adj_pts[:]
		row_num = 0
		# uv_pts = self.uv_pts[row_num]
		uv_pts = copy.deepcopy(self.uv_pts[row_num])
		self.max_y_list = [0,0,0]*len(uv_pts)




		#Boundary Spline for very first spline
		# first_max_y_list = []
		for i in range(len(uv_pts)):
			w, v = self.shape_operator(uv_pts[i])
			print("w: ", w)
			print("v: ", v)
			max_y, min_y = self.ribbon_width(w, v, uv_pts[i], self.plot_ellipse)   #max_y is array
			print("max_y", max_y)
			self.prev_max_y_list.append(max_y.tolist())
			self.prev_min_y_list.append(min_y.tolist())

		# self.prev_max_y_list.pop(0)
		# self.prev_max_y_list.pop(-1)
		print("prev_max_y_list", self.prev_max_y_list)
		# edge_curve = BSpline.Curve() 
		# edge_curve.degree = 5
		# edge_curve.ctrlpts = copy.deepcopy(self.prev_max_y_list)
		# edge_curve.knotvector = utilities.generate_knot_vector(edge_curve.degree, len(edge_curve.ctrlpts))
		# edge_curve.delta = 0.001
		# edge_curve.evaluate()
		# edge_curve_pts = edge_curve.evalpts
		# self.edge_pts.append(edge_curve_pts)
		# # print("edge_curve_pts", edge_curve_pts)

		# edge_curve_arr = np.asarray(edge_curve_pts)

		# self.ax.scatter(edge_curve_arr[:,0], edge_curve_arr[:,1], edge_curve_arr[:,2], color='blue')



		#Continue with remaining splines
		row_num = 1


		# while not self.covered:
		# for k in range(self.num_rows):
		while (self.cover_area_tot + self.uncovered_area_tot - self.overlap_area_tot) < 0.5:
			print("TRAJECTORY ROW: ", row_num)
			for i in range(len(uv_pts)):
				print("UV pt: ", uv_pts[i])
				w, v = self.shape_operator(uv_pts[i])
				x = threading.Thread(target=self.thread_func_equal, args=(uv_pts[i],))
				self.threads.append(x)
				x.start()

			for index, thread in enumerate(self.threads):
				logging.info("Main    : before joining thread %d.", index)
				thread.join()
				logging.info("Main    : thread %d done", index)



			pts_array = np.asarray(self.uv_pts_row)
			new_max_y_array = np.asarray(self.new_max_y_list)
			new_min_y_array = np.asarray(self.new_min_y_list)
			concat_array = np.concatenate((pts_array, new_max_y_array, new_min_y_array), axis=1)
			if self.direction == 'u':
				concat_array = concat_array[np.argsort(concat_array[:, 0])]
			if self.direction == 'v':
				concat_array = concat_array[np.argsort(concat_array[:, 1])]
			pts_array = concat_array[:,0:2]
			new_max_y_array = concat_array[:,2:5]
			new_min_y_array = concat_array[:,5:8]

			prev_max_y_array = np.asarray(self.prev_max_y_list)
			prev_min_y_array = np.asarray(self.prev_min_y_list)

			prev_max_y_array = prev_max_y_array[prev_max_y_array[:,0].argsort()]
			prev_min_y_array = prev_min_y_array[prev_min_y_array[:,0].argsort()]

			# Find Coverage Area

			for i in range(prev_max_y_array.shape[0]-1):
				pt0 = prev_min_y_array[i,:]
				pt1 = prev_max_y_array[i,:]
				pt2 = prev_max_y_array[i+1,:]
				pt3 = prev_min_y_array[i+1,:]
				#New minimum points
				pt4 = new_min_y_array[i,:]
				pt5 = new_min_y_array[i+1,:]
				#Covered area
				cover_pts = np.vstack((pt0, pt1, pt2, pt3))
				cover_area = self.area_slice(cover_pts, self.plot_area, 'b')
				self.cover_area_tot = self.cover_area_tot + cover_area
				

				#Overlap area
				overlap_pts = np.vstack((pt1, pt2, pt5, pt4))
				# print("Overlap Points: ", overlap_pts)
				if pt4[1] > pt1[1]:
					uncovered_area = self.area_slice(overlap_pts, self.plot_area, 'y')
					self.uncovered_area_tot = self.uncovered_area_tot + uncovered_area
				else:
					overlap_area = self.area_slice(overlap_pts, self.plot_area, 'r')
					self.overlap_area_tot = self.overlap_area_tot + overlap_area

			pts_list = pts_array.tolist()
			# print("pts_list", pts_list)


			max_y_list = new_max_y_array.tolist()


			if self.increasing == False:
				max_y_list = max_y_list[::-1]

			self.uv_pts.append(pts_list)
			# print("uv_pts", self.uv_pts)

			# del self.max_y_list[:]
			del uv_pts[:]

			del self.prev_max_y_list[:]
			del self.prev_min_y_list[:]
			self.prev_max_y_list = []
			self.prev_min_y_list = []
			self.prev_max_y_list = copy.deepcopy(self.new_max_y_list)
			self.prev_min_y_list = copy.deepcopy(self.new_min_y_list)
			del self.new_max_y_list[:]
			del self.new_min_y_list[:]
			self.new_max_y_list = []
			self.new_min_y_list = []

			del self.uv_pts_row[:]


			# self.max_y_list = []
			uv_pts = []
			# self.new_max_y_list = []
			self.uv_pts_row = []
			# self.edge_pts = []


			# for i in range(len(max_y_list)):
			# 	self.max_y_list.append(max_y_list[i])

			for i in range(len(pts_list)):
				uv_pts.append(pts_list[i])



			#Form ribbon edge boundary
			# print("max_y_list", max_y_list)
			# edge_curve = BSpline.Curve() 
			# edge_curve.degree = 5
			# edge_curve.ctrlpts = copy.deepcopy(max_y_list)
			# edge_curve.knotvector = utilities.generate_knot_vector(edge_curve.degree, len(edge_curve.ctrlpts))
			# edge_curve.delta = 0.001
			# edge_curve.evaluate()
			# edge_curve_pts = edge_curve.evalpts
			# self.edge_pts.append(edge_curve_pts)
			# # print("edge_curve_pts", edge_curve_pts)

			# edge_curve_arr = np.asarray(edge_curve_pts)

			# self.ax.scatter(edge_curve_arr[:,0], edge_curve_arr[:,1], edge_curve_arr[:,2], color='blue')

			row_num = row_num + 1


			print("concat_array", concat_array)
			print("pts_array", pts_array)
			print("max_y_array", new_max_y_array)
			print("prev_y_array", prev_max_y_array)
			print("min_y_array", new_min_y_array)




	'''
	def base_pts_equal(self):
		# del self.k_1_dir[:]
		# del self.k_2_dir[:]
		del self.adj_pts[:]
		row_num = 0
		# uv_pts = self.uv_pts[row_num]
		uv_pts = copy.deepcopy(self.uv_pts[row_num])
		#self.max_y_list = [0,0,0]*len(uv_pts)


		#Continue with remaining splines
		row_num = 1


		# while not self.covered:
		for k in range(self.num_rows_equal):
			print("TRAJECTORY ROW: ", row_num)
			for i in range(len(uv_pts)):
				y = threading.Thread(target=self.thread_func_equal, args=(uv_pts[i],))
				self.threads_equal.append(y)
				y.start()

			for index, thread in enumerate(self.threads_equal):
				logging.info("Main    : before joining thread %d.", index)
				thread.join()
				logging.info("Main    : thread %d done", index)



			pts_array = np.asarray(self.uv_pts_row)

			if self.direction == 'u':
				pts_array = pts_array[np.argsort(pts_array[:, 0])]
			if self.direction == 'v':
				pts_array = pts_array[np.argsort(pts_array[:, 1])]
			
			pts_list = pts_array.tolist()
			

			self.uv_pts.append(pts_list)
			# print("uv_pts", self.uv_pts)


			del uv_pts[:]

			del self.uv_pts_row[:]
			uv_pts = []

			self.uv_pts_row = []


			for i in range(len(pts_list)):
				uv_pts.append(pts_list[i])

			row_num = row_num + 1
			print("pts_array", pts_array)
	'''

	#Description:  Computes and plots surface area covered by sander
	def area_slice(self, pts, plot, color):
		side_1 = np.linalg.norm(pts[0,:] - pts[1,:])
		side_2 = np.linalg.norm(pts[0,:] - pts[3,:])
		area = side_1*side_2
		# print("Points: ", pts)
		# p0 = pts[0,:].tolist()
		# p1 = pts[1,:].tolist()
		# p2 = pts[2,:].tolist()
		# p3 = pts[3,:].tolist()
		x = pts[:,0].tolist()
		y = pts[:,1].tolist()
		if color == 'r':
			z = (pts[:,2] + 0.06).tolist()
		else:
			z = (pts[:,2] + 0.05).tolist()
		vertices = [list(zip(x, y, z))]
		# vertices = [list(zip(p0, p1, p2, p3))]
		# self.ax.add_collection3d(Poly3DCollection(vertices, linewidths=0, facecolors=[color], alpha=1.0, zorder=100))
		if plot == True:
			if color == 'y':
				self.ax.add_collection3d(Poly3DCollection(vertices, facecolors=[color], alpha=0.0))
			# if color == 'r':
			# 	self.ax.add_collection3d(Poly3DCollection(vertices, facecolors=[color], zorder=100))
			if color == 'b':
				self.ax.add_collection3d(Poly3DCollection(vertices, facecolors=[color], alpha=0.5, zorder=0))
		return area

	# def line_eqn(self, pt1, pt2):
	


	#Description:  Shape operator function for finding curvature and direction of curvature at each point
	def shape_operator(self, uv_pt):
		normal = np.asarray(self.surf.normal(uv_pt))[1,:]
		point = np.asarray(self.surf.normal(uv_pt))[0,:]
		SKL = self.surf.derivatives(uv_pt[0], uv_pt[1], 2)
		E = np.asarray(SKL[1][0]).dot(np.asarray(SKL[1][0]))
		F = np.asarray(SKL[0][1]).dot(np.asarray(SKL[1][0]))
		G = np.asarray(SKL[0][1]).dot(np.asarray(SKL[0][1]))
		g = normal.dot(np.asarray(SKL[0][2]))
		f = normal.dot(np.asarray(SKL[1][2]))
		e = normal.dot(np.asarray(SKL[2][0]))

		# print('point', point)
		# print('norm', normal)
		den = 1/(E*G-F**2)
		I = np.array([[E, F],
					   [F, G]])
		II = -np.array([[e, f],
					   [f, g]])
		S = (II).dot(np.linalg.inv(I))

		# ind = np.argwhere(S == 0)
		K = np.linalg.det(S)
		H = -0.5*(S[0,0] + S[1,1])

		w, v = np.linalg.eig(S)

		w[0] = H + np.sqrt(H**2 - K)
		w[1] = H - np.sqrt(H**2 - K)

		# print("eig_vals: ", w)
		# print("eig_vecs: ", v)
		# print('S', S)
		return w, v

	#Description:  Find cartesian point on ellipse
	def ellipse_maj_pt(self, w_dir, v_dir, uv_pt, param):
		adj_dist = 0.0  #distance between adjacent hatch points in v-direction

		u_curr = uv_pt[0]
		v_curr = uv_pt[1]

		# print("w_dir: ", w_dir)

		duv = np.zeros((2,1))
		init_pt = self.surf.evaluate_single(uv_pt)
		prev_pt = init_pt
		next_pt = self.surf.evaluate_single([u_curr, v_curr])

		if param == "u":
			vec_dir = v_dir[0]
			# string = "black"
		elif param == "v":
			vec_dir = v_dir[1]
			# string = "orange"

		while abs(adj_dist - w_dir) > 1e-3:
			# print("here")
			if u_curr < 0.0 or v_curr < 0.0:
				# print("u_curr: ", u_curr)
				# print("v_curr: ", v_curr)
				next_pt = [np.NAN, np.NAN, np.NAN]
				break
			if u_curr > 1.0 or v_curr > 1.0:
				# print("u_curr: ", u_curr)
				# print("v_curr: ", v_curr)
				# u_curr = 2
				# v_curr = 2
				next_pt = [np.NAN, np.NAN, np.NAN]
				break
			next_pt = self.surf.evaluate_single([u_curr, v_curr])
			next_pt_arr = np.asarray(next_pt)
			prev_pt_arr = np.asarray(prev_pt)
			incr_dist = np.linalg.norm(next_pt_arr - prev_pt_arr)
			adj_dist = adj_dist + incr_dist
			prev_pt = next_pt
			error = adj_dist - w_dir
			# print("HERE 1")
			# print("error: ", error)
			# print("vec_dir: ", vec_dir)
			# k = 1e-3 * error
			k = 1e-2 * error
			if vec_dir < 0:
				# print("opt 1")
				duv = k*v_dir
			else:
				duv = -k*v_dir
				# print("opt 2")
			u_curr = u_curr + duv[0]
			v_curr = v_curr + duv[1]

		init_pt_arr = np.asarray(init_pt)
		next_pt_arr = next_pt
		pts_array = np.vstack((init_pt_arr,next_pt_arr))
		# self.ax.plot(pts_array[:,0], pts_array[:,1], pts_array[:,2], color=string)
		# print("ellipse next pt", next_pt)
		return next_pt

	#Description:  Find the cartesian point with either the maximum x or y coordinate
	def find_ellipse_max(self, uv_pt, w_u, w_v, w_u_cart, w_v_cart, plot):
		# self.surf.evaluate(start_u=uv_pt[0]-0.1, stop_u=uv_pt[0]+0.1, start_v=uv_pt[1]-0.1, stop_v=uv_pt[1]+0.1)
		# surface_points = self.surf.evalpts

		ctr_pt = np.asarray(self.surf.evaluate_single(uv_pt))
		# print("center", ctr_pt)
		# WU = np.asarray(w_u_cart)
		# WV = np.asarray(w_v_cart)
		# # print("w_u", WU)
		# print("w_v", WV)
		vec_1 = np.asarray(w_u_cart) - ctr_pt
		vec_2 = np.asarray(w_v_cart) - ctr_pt

		# print("vec_1: ", vec_1)
		# print("vec_2: ", vec_2)

		all_zeros_1 = not np.any(vec_1)
		all_zeros_2 = not np.any(vec_2)

		if all_zeros_2 or all_zeros_2:
			return np.ones(3), np.ones(3)
		else:

			ellipse_pts = []
			angle = 0
			rot_angle = -self.hatch_angle

			R = np.array([[math.cos(rot_angle), -math.sin(rot_angle), 0.0],
						  [math.sin(rot_angle), math.cos(rot_angle), 0.0],
						  [0.0, 0.0, 1.0]])

			for i in range(100):
				pt = math.cos(angle)*vec_1 + math.sin(angle)*vec_2 #+ ctr_pt
				angle = angle + np.pi/50.0
				point = pt.tolist()
				ellipse_pts.append(point)

			ell = np.asarray(ellipse_pts)


			rot_ell = np.zeros(ell.shape)

			for j in range(ell.shape[0]):        #Rotate Ellipse by negative of hatch angle
				rot_ell[j,:] = R.dot(ell[j,:])

			# if self.direction == 'u':
			# 	ind_max = np.argmax(rot_ell[:,1], axis=0)
			# 	ind_min = np.argmin(rot_ell[:,1], axis=0)

			# if self.direction == 'v':
			# 	ind_max = np.argmax(rot_ell[:,0], axis=0)
			# 	ind_min = np.argmin(rot_ell[:,0], axis=0)

			if self.direction == 'u':
				ind_max = np.argmax(rot_ell[:,0], axis=0)
				ind_min = np.argmin(rot_ell[:,0], axis=0)

			if self.direction == 'v':
				ind_max = np.argmax(rot_ell[:,1], axis=0)
				ind_min = np.argmin(rot_ell[:,1], axis=0)


			for j in range(ell.shape[0]):        #Add back center point
				for k in range(ell.shape[1]):
					ell[j,k] = ell[j,k] + ctr_pt[k] 



			max_y = ell[ind_max, :]
			min_y = ell[ind_min, :]
			# print('max_y', max_y)
			if plot:
				# print("PLOTTING")
				self.ax.scatter(ell[:,0], ell[:,1], ell[:,2], color='blue')
				self.ax.scatter(max_y[0], max_y[1], max_y[2], color='black')
			return max_y, min_y
			


	def ribbon_width(self,w,v,uv_pt, plot):

		if abs(w[0]) < 1e-9: #6e-17:   #u-direction
			p_u = np.inf
			# print("here 2")
		elif w[0] > 0:
			p_u = 1.0/w[0]
		else:
			p_u = -1.0/w[0]
		w_u = 2*np.sqrt(p_u**2 -(p_u - self.ecc)**2) #euclidean contact width
		if w_u > self.radius or math.isnan(w_u):
			w_u = self.radius

		# print("w[1]: ", w[1])

		if abs(w[1]) < 1e-9: #6e-17:    #v-direction
			p_v = np.inf
			# print("here 2")
		elif w[1] > 0:
			p_v = 1.0/w[1]
		else:
			p_v = -1.0/w[1]
		w_v = 2*np.sqrt(p_v**2 -(p_v - self.ecc)**2) #euclidean contact width
		# print("first w_v: ", w_v)
		if w_v > self.radius or math.isnan(w_v):
			print("MAX WIDTH")
			w_v = self.radius

		# print("v: ", v)
		# print("w_u: ", w_u)
		# print("w_v: ", w_v)

		if self.direction == 'u':
			# print('U_dir')
			w_v_cart = self.ellipse_maj_pt(w_v, v[:,1], uv_pt, "v")
			w_u_cart = self.ellipse_maj_pt(w_u, v[:,0], uv_pt, "u")

		if self.direction == 'v':
			# print('V_dir')
			w_v_cart = self.ellipse_maj_pt(w_u, v[:,1], uv_pt, "v")
			w_u_cart = self.ellipse_maj_pt(w_v, v[:,0], uv_pt, "u")

		# print("w_u_cart: ", w_u_cart)
		# print("w_v_cart: ", w_v_cart)

		if any(w_v_cart) == np.NAN or any(w_u_cart) == np.NAN:
			max_y = [np.NAN, np.NAN, np.NAN]
			min_y = [np.NAN, np.NAN, np.NAN]
		else:
			max_y, min_y = self.find_ellipse_max(uv_pt, w_u, w_v, w_u_cart, w_v_cart, plot)
		return max_y, min_y

	def avg_curvature(self, pt_c):
		num_samples = 50
		sum_v = np.zeros((2,2))
		sum_w = np.zeros(2)
		pt_test = np.empty(2)
		for i in range(num_samples):
			rand_u = random.uniform(-0.01,0.01)
			rand_v = random.uniform(-0.01,0.01)
			pt_test[0] = pt_c[0] + rand_u
			pt_test[1] = pt_c[1] + rand_v
			pt_test_list = pt_test.tolist()
			W_test, V_test = self.shape_operator(pt_test_list)
			sum_w = np.add(sum_w, W_test)
			sum_v = np.add(sum_v, V_test)

		W_avg = sum_w / num_samples
		V_avg = sum_v / num_samples
		return W_avg, V_avg
		

	def avg_normal(self, pt_c):
		num_samples = 50
		sum_norm = np.zeros(3)
		pt_test = np.empty(2)

		print("pt_c:  ", pt_c)
		
		for i in range(num_samples):
			norm_samp_list = []
			rand_u = random.uniform(-0.01,0.01)
			rand_v = random.uniform(-0.01,0.01)
			pt_test[0] = pt_c[0] + rand_u
			pt_test[1] = pt_c[1] + rand_v
			pt_test_list = pt_test.tolist()

			norm_samp = self.surf.normal(pt_test_list)
			# print("norm_samp: ", norm_samp)
			# print("norm_samp_1: ", norm_samp[0][1])
			# print("norm_samp_2: ", norm_samp[1][0])
			# print("norm_len: ", len(norm_samp[1]))

			for j in range(len(norm_samp[1])): 
				norm_samp_list.append(norm_samp[1][j])
			norm_arr = np.asarray(norm_samp_list)
			sum_norm = np.add(sum_norm, norm_arr)
			del norm_samp_list[:]

		avg_norm = sum_norm/num_samples
		avg_norm = avg_norm/np.linalg.norm(avg_norm)  #normalize
		avg_norm_list = avg_norm.tolist()

		return avg_norm_list



	def adjacent_pt(self, uv_pt, edge_pts, i):
		fc = 100
		u_init = uv_pt[0]
		v_init = uv_pt[1]
		out_of_bounds = False
		a = -0.05
		b = 0.4   #need method to initialize this number
		edge_y = [0.0,0.0,0.0]
		first_iter = True 
		max_iter = 50
		iter_num = 0


		if self.direction == 'u':


			while abs(fc) > 1e-3 and iter_num < max_iter:   #Bisection Method
				c = (a+b)/2.0
				v_c = v_init + c
				pt_c = [u_init, v_c]
				if v_c > 1.0:
					pt_c = [u_init, 1.0]
					break
				W_c, V_c = self.shape_operator(pt_c)
				# W_c, V_c = self.avg_curvature(pt_c)
				# print("W_c: ", W_c)
				# print("V_c: ", V_c)

				# if pt_c[0] > 0.82 and pt_c[0] < 0.84:
				# 	print("W_c Normal: ", W_c)
				# if pt_c[0] > 0.87 and pt_c[0] < 0.89:
				# 	print("W_c Error: ", W_c)


				max_c, min_c = self.ribbon_width(W_c,V_c,pt_c, False)
				if first_iter == True:  #Find corresponding location on boundary edge
					if min_c[0] <= edge_pts[0][0]:
						edge_y = edge_pts[0]
					elif min_c[0] >= edge_pts[-1][0]:
						edge_y = edge_pts[-1]
					else:
						for i in range(len(edge_pts)):
							if edge_pts[i][0] >= min_c[0]:
								edge_y = edge_pts[i]
								first_iter = False
								break
					# print("")

				fc = edge_y[1] - min_c[1]
				# print("thread num: ", i)
				# print("edge pt: ", edge_y)
				# print("min_c", min_c)
				# print("error", fc)
				v_a = v_init + a
				pt_a = [u_init, v_a]
				W_a, V_a = self.shape_operator(pt_a)
				max_a, min_a = self.ribbon_width(W_a,V_a,pt_a, False)
				if np.isnan(min_a).any():
					out_of_bounds = True
					break
				fa = edge_y[1] - min_a[1]
				if np.sign(fc) == np.sign(fa):
					a = c
				else:
					b = c

				iter_num = iter_num + 1

			if out_of_bounds == True:
				print("Point Out of Bounds")
				pt_c = [0,0]
				new_max_y = [np.NAN, np.NAN, np.NAN]
			else:
				print("Adjacent Point Found")
				w, v = self.shape_operator(pt_c)
				new_max_y, new_min_y = self.ribbon_width(w,v,pt_c,self.plot_ellipse)


		if self.direction == 'v':

			while abs(fc) > 1e-3 and iter_num < max_iter:   #Bisection Method
				c = (a+b)/2.0
				u_c = u_init + c
				pt_c = [u_c, v_init]
				if u_c > 1.0:
					pt_c = [1.0, v_init]
					break
				W_c, V_c = self.shape_operator(pt_c)
				# W_c, V_c = self.avg_curvature(pt_c)
				# print("W_c: ", W_c)
				# print("V_c: ", V_c)


				# if pt_c[0] > 0.82 and pt_c[0] < 0.84:
				# 	print("W_c Normal: ", W_c)
				# if pt_c[0] > 0.87 and pt_c[0] < 0.89:
				# 	print("W_c Error: ", W_c)


				max_c, min_c = self.ribbon_width(W_c,V_c,pt_c, False)

				if first_iter == True:  #Find corresponding location on boundary edge
					if min_c[0] <= edge_pts[0][0]:
						edge_y = edge_pts[0]
					elif min_c[0] >= edge_pts[-1][0]:
						edge_y = edge_pts[-1]
					else:
						for i in range(len(edge_pts)):
							if edge_pts[i][0] >= min_c[0]:
								edge_y = edge_pts[i]
								first_iter = False
								break
					# print("")

				fc = edge_y[1] - min_c[1]
				# print("thread num: ", i)
				# print("edge pt: ", edge_y)
				# print("min_c", min_c)
				# print("error", fc)
				u_a = u_init + a
				pt_a = [u_a, v_init]
				W_a, V_a = self.shape_operator(pt_a)
				max_a, min_a = self.ribbon_width(W_a,V_a,pt_a, False)

				if i == 4: 
					print("thread num: ", i)
					print("edge pt: ", edge_y)
					print("min_c", min_c)
					print("max_c: ", max_c)
					print("error", fc)
					print("min_a: ", min_a)
					print("\n")

				if i == 5: 
					print("thread num: ", i)
					print("edge pt: ", edge_y)
					print("min_c", min_c)
					print("max_c: ", max_c)
					print("error", fc)
					print("min_a: ", min_a)
					print("\n")

				if i == 7: 
					print("thread num: ", i)
					print("edge pt: ", edge_y)
					print("min_c", min_c)
					print("max_c: ", max_c)
					print("error", fc)
					print("min_a: ", min_a)
					print("\n")

				if np.isnan(min_a).any():
					out_of_bounds = True
					break
				fa = edge_y[1] - min_a[1]
				if np.sign(fc) == np.sign(fa):
					a = c
					# print("here 1")
				else:
					b = c
					# print("here 2")

				iter_num = iter_num + 1

			if out_of_bounds == True:
				print("Point Out of Bounds")
				pt_c = [0,0]
				new_max_y = [np.NAN, np.NAN, np.NAN]
			else:
				print("Adjacent Point Found")
				w, v = self.shape_operator(pt_c)
				new_max_y, new_min_y = self.ribbon_width(w,v,pt_c,self.plot_ellipse)


		return pt_c, new_max_y, new_min_y




	def adjacent_pt_equal(self, uv_pt):
	
		adj_dist = 0.0  #distance between adjacent hatch points in v-direction

		u_curr = uv_pt[0]
		v_curr = uv_pt[1]

		# print("w_dir: ", w_dir)

		init_pt = self.surf.evaluate_single(uv_pt)
		prev_pt = init_pt
		next_pt = init_pt


		while abs(adj_dist - self.equal_spacing) > 1e-3:
			# print("here")
			# print("u_curr: ", u_curr)
			if u_curr < 0.0 or v_curr < 0.0:
				print("u_curr: ", u_curr)
				print("v_curr: ", v_curr)
				next_pt = [np.NAN, np.NAN, np.NAN]
				break
			if u_curr > 1.0 or v_curr > 1.0:

				next_pt = [np.NAN, np.NAN, np.NAN]
				break
			next_pt = self.surf.evaluate_single([u_curr, v_curr])
			next_pt_arr = np.asarray(next_pt)
			prev_pt_arr = np.asarray(prev_pt)

			'''

			if self.direction == 'u':
				incr_dist = abs(next_pt_arr[1] - prev_pt_arr[1])

			if self.direction == 'v':
				incr_dist = abs(next_pt_arr[0] - prev_pt_arr[0])
				# print("incr_dist: ", incr_dist)
			'''

			incr_dist = abs(np.linalg.norm(next_pt_arr - prev_pt_arr))


			adj_dist = adj_dist + incr_dist
			# print("adj: ", adj_dist)
			# print("incr dist: ", incr_dist)
			prev_pt = next_pt
			error = adj_dist - self.equal_spacing
			# print("error: ", error)
			# print("vec_dir: ", vec_dir)
			# k = -1e-5 * error
			k = -1e-2 * error
			# print("k: ", k)
			if self.direction == 'u':
				v_curr = v_curr + k

			if self.direction == 'v':
				u_curr = u_curr + k
				# print('U_curr: ', u_curr)
			
		pt_c = [u_curr, v_curr]
		# print("equal pt: ", pt_c)
		w, v = self.shape_operator(pt_c)
		new_max_y, new_min_y = self.ribbon_width(w,v,pt_c,self.plot_ellipse)

		return pt_c, new_max_y, new_min_y


	def minimum_ellipse(self):
		min_dist = np.inf
		for i in range(len(self.uv_pts)-1):      #use original UV pts for bisection search of way points
			self.curve_data_pts = []
			for j in range(len(self.uv_pts[i])):
				end_pt = self.surf.evaluate_single(self.uv_pts[i+1][j])
				start_pt = self.surf.evaluate_single(self.uv_pts[i][j])
				if self.direction == 'u':
					dist = abs(end_pt[0] - start_pt[0])
				if self.direction == 'v':
					dist = abs(end_pt[1] - start_pt[1])
				if dist < min_dist:
					min_dist = dist
		print("MIN DIST: ", min_dist)
		self.equal_spacing = min_dist


	def calc_equal_num_rows(self):
		end_pt = self.surf.evaluate_single(self.uv_pts[-1][0])
		start_pt = self.surf.evaluate_single(self.uv_pts[0][0])
		if self.direction == 'u':
			dist = abs(end_pt[0] - start_pt[0])
		if self.direction == 'v':
			dist = abs(end_pt[1] - start_pt[1])
		self.num_rows_equal = int(dist/self.equal_spacing)
		print("Num Equal Rows: ", self.num_rows_equal)


	def scan_splines(self):
		# print("UV_PTS: ", self.uv_pts)
		for i in range(len(self.uv_pts)):      #use original UV pts for bisection search of way points
			self.curve_data_pts = []
			for j in range(len(self.uv_pts[i])):
				pt = self.surf.evaluate_single(self.uv_pts[i][j])
				# print("surf_pt", pt)
				self.curve_data_pts.append(pt)
			# print("curve data pts: ", self.curve_data_pts)
			curve = BSpline.Curve() 
			curve.degree = 4
			# print("curve_pts", self.curve_data_pts)
			cart_row = copy.deepcopy(self.curve_data_pts)
			self.cart_pts.append(cart_row)
			curve.ctrlpts = copy.deepcopy(self.curve_data_pts)
			curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))
			curve.delta = 0.001
			curve.evaluate()
			curve_length = operations.length_curve(curve) #Find curve length and add to total
			self.traj_total_length = self.traj_total_length + curve_length
			curve_points = curve.evalpts
			data = np.asarray(curve_points)
			self.ax.plot(data[:,0],data[:,1],data[:,2], color='red', linewidth=3)  #plot splines
			self.ax1.plot(data[:,0],data[:,1],data[:,2], color='red', linewidth=3)  #plot splines
			way_pts_row = self.gen_spline_pts(curve_points)
			# print("way pts row: ", way_pts_row)
			self.scan_way_pts.append(way_pts_row)
			del self.curve_data_pts[:]

		# print("cart pts: ", self.cart_pts)

		# for i in range(len(self.scan_way_pts)):    #plot way points
		# 	for j in range(len(self.scan_way_pts[i])):
		# 		data1 = self.scan_way_pts[i][j]
		# 		data1_arr = np.asarray(data1)
		# 		# print("traj_pts: ", data1_arr)
		# 		self.ax.scatter(data1_arr[0],data1_arr[1],data1_arr[2], c = "r")
		# 		self.ax1.scatter(data1_arr[0],data1_arr[1],data1_arr[2], c = "r")



	def cross_splines(self):
		#Cross-Over Splines
		for i in range(len(self.uv_pts)-1):
			self.cross_data_pts = []
			if i % 2 == 0:
				start_pt = np.asarray(self.uv_pts[i][-1])
				end_pt = np.asarray(self.uv_pts[i+1][-1])
			else:
				start_pt = np.asarray(self.uv_pts[i][0])
				end_pt = np.asarray(self.uv_pts[i+1][0])

			u_pts = np.linspace(start_pt[0], end_pt[0], 20)[:, np.newaxis]
			v_pts = np.linspace(start_pt[1], end_pt[1], 20)[:, np.newaxis]
			uv_pts = np.concatenate((u_pts, v_pts), axis=1)
			uv_pts_list = uv_pts.tolist()
			for j in range(len(uv_pts_list)):
				pt = self.surf.evaluate_single(uv_pts_list[j])
				# print("surf_pt", pt)
				self.cross_data_pts.append(pt)
			# print("curve data pts: ", self.curve_data_pts)
			curve = BSpline.Curve() 
			curve.degree = 4
			cart_row = copy.deepcopy(self.cross_data_pts)

			self.cross_cart_pts.append(cart_row)
			curve.ctrlpts = copy.deepcopy(self.cross_data_pts)
			curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))
			curve.delta = 0.001
			curve.evaluate()
			curve_length = operations.length_curve(curve) #Find curve length and add to total
			self.traj_total_length = self.traj_total_length + curve_length
			curve_points = curve.evalpts
			data = np.asarray(curve_points)
			self.ax.plot(data[:,0],data[:,1],data[:,2], color='red', linewidth=3)  #plot splines
			self.ax1.plot(data[:,0],data[:,1],data[:,2], color='red', linewidth=3)  #plot splines
			way_pts_row = self.gen_spline_pts(curve_points)
			# print("way pts row: ", way_pts_row)
			self.cross_way_pts.append(way_pts_row)
			del self.cross_data_pts[:]

		self.cross_way_pts.append([])     #append extra row to match length of scan splines

		# for i in range(len(self.cross_way_pts)):    #plot way points
		# 	for j in range(len(self.cross_way_pts[i])):
		# 		data1 = self.cross_way_pts[i][j]
		# 		data1_arr = np.asarray(data1)
		# 		# print("traj_pts: ", data1_arr)
		# 		self.ax.scatter(data1_arr[0],data1_arr[1],data1_arr[2], c = "r")
		# 		self.ax1.scatter(data1_arr[0],data1_arr[1],data1_arr[2], c = "r")
		# print("cross cart: ", self.cross_cart_pts)


	def scan_splines_equal(self):
		# print("UV_PTS: ", self.uv_pts)
		for i in range(len(self.uv_pts)):      #use original UV pts for bisection search of way points
			self.curve_data_pts = []
			for j in range(len(self.uv_pts[i])):
				pt = self.surf.evaluate_single(self.uv_pts[i][j])
				# print("surf_pt", pt)
				self.curve_data_pts.append(pt)
			# print("curve data pts: ", self.curve_data_pts)
			curve = BSpline.Curve() 
			curve.degree = 4
			# print("curve_pts", self.curve_data_pts)
			cart_row = copy.deepcopy(self.curve_data_pts)
			self.cart_pts.append(cart_row)
			curve.ctrlpts = copy.deepcopy(self.curve_data_pts)
			curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))
			curve.delta = 0.001
			curve.evaluate()
			curve_length = operations.length_curve(curve) #Find curve length and add to total
			self.traj_total_length = self.traj_total_length + curve_length
			curve_points = curve.evalpts
			data = np.asarray(curve_points)
			self.ax.plot(data[:,0],data[:,1],data[:,2], color='red', linewidth=3)  #plot splines
			self.ax1.plot(data[:,0],data[:,1],data[:,2], color='red', linewidth=3)  #plot splines
			way_pts_row = self.gen_spline_pts(curve_points)
			# print("way pts row: ", way_pts_row)
			self.scan_way_pts.append(way_pts_row)
			del self.curve_data_pts[:]
		# print("Scan Way Points: ", self.scan_way_pts)

		# print("cart pts: ", self.cart_pts)

		# for i in range(len(self.scan_way_pts)):    #plot way points
		# 	for j in range(len(self.scan_way_pts[i])):
		# 		data1 = self.scan_way_pts[i][j]
		# 		data1_arr = np.asarray(data1)
		# 		# print("traj_pts: ", data1_arr)
		# 		self.ax.scatter(data1_arr[0],data1_arr[1],data1_arr[2], c = "b")
		# 		self.ax1.scatter(data1_arr[0],data1_arr[1],data1_arr[2], c = "b")



	def cross_splines_equal(self):
		#Cross-Over Splines
		for i in range(len(self.uv_pts)-1):
			self.cross_data_pts = []
			if i % 2 == 0:
				start_pt = np.asarray(self.uv_pts[i][-1])
				end_pt = np.asarray(self.uv_pts[i+1][-1])
			else:
				start_pt = np.asarray(self.uv_pts[i][0])
				end_pt = np.asarray(self.uv_pts[i+1][0])

			u_pts = np.linspace(start_pt[0], end_pt[0], 20)[:, np.newaxis]
			v_pts = np.linspace(start_pt[1], end_pt[1], 20)[:, np.newaxis]
			uv_pts = np.concatenate((u_pts, v_pts), axis=1)
			uv_pts_list = uv_pts.tolist()
			for j in range(len(uv_pts_list)):
				pt = self.surf.evaluate_single(uv_pts_list[j])
				# print("surf_pt", pt)
				self.cross_data_pts.append(pt)
			# print("curve data pts: ", self.curve_data_pts)
			curve = BSpline.Curve() 
			curve.degree = 4
			cart_row = copy.deepcopy(self.cross_data_pts)

			self.cross_cart_pts.append(cart_row)
			curve.ctrlpts = copy.deepcopy(self.cross_data_pts)
			curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))
			curve.delta = 0.001
			curve.evaluate()
			curve_length = operations.length_curve(curve) #Find curve length and add to total
			self.traj_total_length = self.traj_total_length + curve_length
			curve_points = curve.evalpts
			data = np.asarray(curve_points)
			self.ax.plot(data[:,0],data[:,1],data[:,2], color='red', linewidth=3)  #plot splines
			self.ax1.plot(data[:,0],data[:,1],data[:,2], color='red', linewidth=3)  #plot splines
			way_pts_row = self.gen_spline_pts(curve_points)
			# print("way pts row: ", way_pts_row)
			self.cross_way_pts.append(way_pts_row)
			del self.cross_data_pts[:]

		self.cross_way_pts.append([])     #append extra row to match length of scan splines

		# for i in range(len(self.cross_way_pts)):    #plot way points
		# 	for j in range(len(self.cross_way_pts[i])):
		# 		data1 = self.cross_way_pts[i][j]
		# 		data1_arr = np.asarray(data1)
		# 		# print("traj_pts: ", data1_arr)
		# 		self.ax.scatter(data1_arr[0],data1_arr[1],data1_arr[2], c = "b")
		# 		self.ax1.scatter(data1_arr[0],data1_arr[1],data1_arr[2], c = "b")



	def gen_spline_pts(self, curve_pts):
		pts = np.asarray(curve_pts)
		traj_pts = []
		i = 0
		init_i = 0
		sum_dist = 0.0
		init_pt = pts[0,:].tolist()  #append initial point
		traj_pts.append(init_pt)
		while i < pts.shape[0]:
			# dist = np.linalg.norm(pts[i,:]-pts[init_i,:])
			dist = np.sqrt((pts[i,0]-pts[init_i,0])**2 + (pts[i,1]-pts[init_i,1])**2 + (pts[i,2]-pts[init_i,2])**2)
			sum_dist = sum_dist + dist 
			# print("sum_dist: ", sum_dist)
			if sum_dist > self.wp_spacing:
				# print("Add traj pt")
				way_pt = pts[i,:].tolist()
				traj_pts.append(way_pt)
				sum_dist = 0.0
				init_i = i
			else:
				i = i + 1
		# print("Num Traj Pts: ", len(traj_pts))
		return traj_pts

	def scan_spline_normals(self):
		for i in range(len(self.scan_way_pts)):
			normal_row = []
			for j in range(len(self.scan_way_pts[i])):
				way_pt = self.scan_way_pts[i][j]
				normal = self.scan_normal_search(way_pt, i)
				normal_row.append(normal)
			norms = copy.deepcopy(normal_row)
			self.scan_normals.append(norms)
			del normal_row[:]

		# for i in range(len(self.scan_normals)):   #parse normal tuples
		# 	for j in range(len(self.scan_normals[i])): 
		# 		self.scan_normals[i][j] = list(self.scan_normals[i][j][1])



	def cross_spline_normals(self):
		for i in range(len(self.cross_way_pts)):
			normal_row = []
			for j in range(len(self.cross_way_pts[i])):
				way_pt = self.cross_way_pts[i][j]
				normal = self.cross_normal_search(way_pt, i)
				normal_row.append(normal)
			norms = copy.deepcopy(normal_row)
			self.cross_normals.append(norms)
			del normal_row[:]

		self.cross_normals.append([])  #append extra row to match scan normals

		# for i in range(len(self.cross_normals)):   #parse normal tuples
		# 	for j in range(len(self.cross_normals[i])): 
		# 		self.cross_normals[i][j] = list(self.cross_normals[i][j][1])



	def scan_normal_search(self, way_pt, ind):
		cart_pts = self.cart_pts[ind]
		uv_pts = self.uv_pts[ind]
		loc = 0
		print("Way Point: ", way_pt)
		# print("Cart Points: ", cart_pts)
		for j in range(len(cart_pts)):
			# print("waypt: ", way_pt[0])
			# print("cart_pt: ", cart_pts[j][0])
			if cart_pts[j][0] >= way_pt[0]:
				# start_uv = uv_pts[j]
				# loc = copy.deepcopy(j)
				break
		# print("j", j)
		start_uv = uv_pts[j]
		curr_cart_pt = self.surf.evaluate_single(start_uv)
		print("start point: ", curr_cart_pt)
		u_curr = start_uv[0]
		v_curr = start_uv[1]

		err_u = 100
		err_v = 100

		# while abs(err_u) > 1e-3:
		# 	if u_curr < 0:
		# 		u_curr = -1
		# 		break
		# 	if u_curr > 1:
		# 		u_curr = 2
		# 		break
		# 	curr_pt = [u_curr, v_curr]
		# 	curr_cart_pt = self.surf.evaluate_single(curr_pt)

		# 	if self.direction == 'u':                  #####COPY and test in other section
		# 		err_u = way_pt[0] - curr_cart_pt[0]
		# 	if self.direction == 'v':
		# 		err_u = way_pt[1] - curr_cart_pt[1]


		# 	print("err_u: ", err_u)
		# 	print("u_curr: ", u_curr)
		# 	du = 1e-2*err_u
		# 	u_curr = u_curr + du


		while abs(err_u) > 1e-3 or abs(err_v) > 1e-3:
			if u_curr < 0 or v_curr < 0:
				u_curr = -1
				v_curr = -1
				break
			if u_curr > 1 or v_curr > 1:
				u_curr = 2
				v_curr = 2
				break
			curr_pt = [u_curr, v_curr]
			curr_cart_pt = self.surf.evaluate_single(curr_pt)
			if self.direction == 'u':                  #####COPY and test in other section
				err_u = way_pt[0] - curr_cart_pt[0]
				err_v = way_pt[1] - curr_cart_pt[1]
			if self.direction == 'v':
				err_u = way_pt[1] - curr_cart_pt[1]
				err_v = way_pt[0] - curr_cart_pt[0]
			# err_u = way_pt[0] - curr_cart_pt[0]
			# err_v = way_pt[1] - curr_cart_pt[1]
			# print("err_u: ", err_u)
			# print("err_v: ", err_v)
			du = 1e-2*err_u
			dv = -1e-2*err_v
			u_curr = u_curr + du
			v_curr = v_curr + dv

		cart_pt_arr = np.asarray(curr_cart_pt)

		# normal = self.surf.normal(curr_pt)
		normal = self.avg_normal(curr_pt)

		# self.ax.scatter(cart_pt_arr[0],cart_pt_arr[1],cart_pt_arr[2], color='green')
		return normal


	def cross_normal_search(self, way_pt, ind):
		cart_pts = self.cross_cart_pts[ind]
		# uv_pts = self.uv_pts[ind]
		if ind % 2 == 0:
			start_uv = self.uv_pts[ind][0]
		else:
			start_uv = self.uv_pts[ind][-1]

		curr_cart_pt = self.surf.evaluate_single(start_uv)
		# print("start point: ", curr_cart_pt)
		u_curr = start_uv[0]
		v_curr = start_uv[1]

		err_u = 100
		err_v = 100

		while abs(err_u) > 1e-3 or abs(err_v) > 1e-3:
			if u_curr < 0 or v_curr < 0:
				u_curr = -1
				v_curr = -1
				break
			if u_curr > 1 or v_curr > 1:
				u_curr = 2
				v_curr = 2
				break
			curr_pt = [u_curr, v_curr]
			curr_cart_pt = self.surf.evaluate_single(curr_pt)
			if self.direction == 'u':                  #####COPY and test in other section
				err_u = way_pt[0] - curr_cart_pt[0]
				err_v = way_pt[1] - curr_cart_pt[1]
			if self.direction == 'v':
				err_u = way_pt[1] - curr_cart_pt[1]
				err_v = way_pt[0] - curr_cart_pt[0]
			# err_u = way_pt[0] - curr_cart_pt[0]
			# err_v = way_pt[1] - curr_cart_pt[1]
			# print("err_u: ", err_u)
			# print("err_v: ", err_v)
			du = 1e-2*err_u
			dv = -1e-2*err_v
			u_curr = u_curr + du
			v_curr = v_curr + dv

		cart_pt_arr = np.asarray(curr_cart_pt)

		# normal = self.surf.normal(curr_pt)
		normal = self.avg_normal(curr_pt)

		# self.ax.scatter(cart_pt_arr[0],cart_pt_arr[1],cart_pt_arr[2], color='green')
		return normal


	def export_traj(self, filename):
		# print("scan wps: ", self.scan_way_pts)
		for i in range(len(self.scan_way_pts)):
			if i%2 == 0:
				pos = self.scan_way_pts[i]
				# print("pos forward: ", pos)
				norm = self.scan_normals[i]
			else:
				pos = (self.scan_way_pts[i])[::-1]
				# print("pos backward: ", pos)
				norm = (self.scan_normals[i])[::-1]

			# print('pos: ', pos)

			for j in range(len(pos)):
				if i == 0 and j == 0:
					f = open(filename,'w+')
					f.truncate()
					f.close()
				f = open(filename,'a')
				pos_arr = np.asarray(pos[j])
				norm_arr = np.asarray(norm[j])
				row = np.concatenate((pos_arr, norm_arr))
				for k in range(row.shape[0]):
					f.write(str(row[k]))
					if k < row.shape[0]-1:
						f.write(',')
				f.write("\n")
				f.close()

			if self.cross_way_pts[i]:
				cross_pos = self.cross_way_pts[i]
				cross_norm = self.cross_normals[i]

				for j in range(len(cross_pos)):
					f = open(filename,'a')
					pos_arr = np.asarray(cross_pos[j])
					norm_arr = np.asarray(cross_norm[j])
					row = np.concatenate((pos_arr, norm_arr))
					for i in range(row.shape[0]):
						f.write(str(row[i]))
						if i < row.shape[0]-1:
							f.write(',')
					f.write("\n")
					f.close()

	# def approach_pts(self):
	# 	#Start and Force pts
	# 	start_pt = np.asarray(self.scan_way_pts[0][0])
	# 	start_norm = np.asarray(self.scan_normals[0][0])
	# 	traj_start_pt = np.add(start_pt, self.first_last_offset*start_norm)
	# 	traj_force_pt = np.add(start_pt, self.force_pt_offset*start_norm)

	# 	#Final Point
	# 	j = len(self.scan_way_pts)-1
	# 	if j%2 == 0:
	# 		end_pt = self.scan_way_pts[j][0]
	# 		# print("pos forward: ", pos)
	# 		end_norm = self.scan_normals[j][0]
	# 	else:
	# 		end_pt = self.scan_way_pts[j][-1]
	# 		# print("pos forward: ", pos)
	# 		end_norm = self.scan_normals[j][-1]

	# 	traj_end_pt = np.add(end_pt, self.first_last_offset*end_norm)
	



	def setup_surface(self):
		self.surf_dim()
		# self.smooth_surface()
		# u = np.array([0.17]) 
		self.plot_surface()

	def uniform_traj_exe(self):
		filename = '/home/daniel/thesis_ws/src/traj_exe/src/data/curved_traj.csv'
		for i in range(self.u_start.shape[0]):
			self.seed_curve(self.u_start[i])
			self.base_pts()
		self.scan_splines()
		self.cross_splines()
		self.scan_spline_normals()
		self.cross_spline_normals()
		self.export_traj(filename)

		# self.minimum_ellipse()
		# self.calc_equal_num_rows()
		print("Traj Length: ", self.traj_total_length)
		self.traj_total_length = 0

	def equal_traj_exe(self):
		filename = '/home/daniel/thesis_ws/src/traj_exe/src/data/standard_traj.csv'
		for i in range(self.u_start.shape[0]):
			self.seed_curve(self.u_start[i])
			self.base_pts_equal()
		self.scan_splines_equal()
		self.cross_splines_equal()
		self.scan_spline_normals()
		self.cross_spline_normals()
		self.export_traj(filename)
		print("Traj Length: ", self.traj_total_length)
	

	def clear_lists(self):
		del self.uv_pts[:]
		del self.cart_pts[:]
		del self.cross_uv_pts[:]
		del self.cross_cart_pts[:]
		del self.uv_pts_row[:]
		del self.cross_way_pts[:]
		del self.scan_way_pts[:]
		del self.scan_normals[:]
		del self.cross_normals[:]

		self.uv_pts = []
		self.cart_pts = []
		self.cross_uv_pts = []
		self.cross_cart_pts = []
		self.uv_pts_row = []
		self.cross_way_pts = []
		self.scan_way_pts = []
		self.scan_normals = []
		self.cross_normals = []


	def execute(self):
		self.setup_surface()
		# self.uniform_traj_exe()

		# self.clear_lists()

		# self.equal_traj_exe()
		print("Covered Area: ", self.cover_area_tot)
		print("Uncovered Area: ", self.uncovered_area_tot)
		print("Overlap Area: ", self.overlap_area_tot)
		print("Non-Overlap Area: ", self.cover_area_tot - self.overlap_area_tot)


if __name__ == '__main__':
	exe = Traj_Gen()
	exe.execute()

	blue_line = mlines.Line2D([], [], color='blue', linewidth =3, label='Standard')
	red_line = mlines.Line2D([], [], color='red', linewidth =3, label='Uniform')

	plt.legend(handles=[blue_line, red_line])
	
	exe.ax.set_xlabel('$x$', fontsize=30)
	exe.ax.set_ylabel('$y$', fontsize=30)
	# exe.ax.legend(handles=[blue_line, red_line])
	# exe.ax.legend(['Uniform', 'Standard'])
	exe.ax.view_init(azim=0, elev=90)

	exe.ax1.set_xlabel('$x$', fontsize=30)
	exe.ax1.set_ylabel('$y$', fontsize=30)
	# exe.ax1.legend(['Uniform', 'Standard'])
	# exe.ax1.legend(handles=[blue_line, red_line])
	exe.ax1.view_init(azim=0, elev=90)

	exe.ax2.set_xlabel('$x$', fontsize=30)
	exe.ax2.set_ylabel('$y$', fontsize=30)
	exe.ax2.view_init(azim=0, elev=90)

	plt.show()