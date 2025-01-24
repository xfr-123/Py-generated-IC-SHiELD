#Author: Mingfei Ren
#Date: 2024-10-26
#Version: 1.0
#Description: This file is used to generate the basic state of the atmosphere, which is the zonal wind field

import numpy as np
from math import factorial
from scipy.special import binom, gamma


class basic_state_generator(): #EC test
    def __init__(self, b = 2.0, n = 3, up = 1.0, gamma = 5e-3):
        self.u0 = 35.0 #reference zonal wind speed, define the zonal mean speed of the jet in troposphere
        self.b = b #nondimensional parameter representing the depth of the jet
        self.n = n #positive integer defining the width of the jet
        self.up = up #amplitude of the perturbation
        self.a = 6371.2e3  #earth radius
        self.g = 9.80616 #gravity acceleration
        self.Rd = 287.05 #gas constant for dry air
        self.omega = 7.2921e-5
        self.eta_t = 0.2 #tropopause level
        self.gamma = gamma #lapse rate
        self.R = self.a / 10 
        self.T_v0 = 288.0 #reference virtual temperature
        self.eta_level_accordingly = self.eta_level_accordingly = np.array([
                                                                0.00100, 0.00395, 0.00808, 0.01361, 0.02064,
                                                                0.02944, 0.04067, 0.05505, 0.07328, 0.09602,
                                                                0.12387, 0.15743, 0.19729, 0.24395, 0.29786,
                                                                0.35909, 0.42660, 0.49748, 0.56756, 0.63328,
                                                                0.69280, 0.74560, 0.79178, 0.83167, 0.86578,
                                                                0.89472, 0.91911, 0.93955, 0.95662, 0.97079,
                                                                0.98256, 0.99223
                                                            ], dtype=np.float32)
        
        '''
        np.array([
            0.00100, 0.00400, 0.00819, 0.01379, 0.02092, 0.02984, 0.04122, 0.05579,
            0.07420, 0.09705, 0.12497, 0.15855, 0.19840, 0.24503, 0.29889, 0.36004,
            0.42746, 0.49824, 0.56822, 0.63384, 0.69327, 0.74600, 0.79210, 0.83192,
            0.86598, 0.89487, 0.91923, 0.93964, 0.95667, 0.97083, 0.98257, 0.99223
        ])
        '''
        '''
        np.array([0.0, 0.00413128, 0.00966086, 0.01669672, 0.02549855, 0.03673121, 
                                      0.05111494, 0.06928385, 0.09183093, 0.11938442, 0.15252788, 0.19185386,
                                      0.23787391, 0.29103042, 0.35138178, 0.41791733, 0.48777655, 0.55683775, 
                                      0.62159457, 0.68024712, 0.73228272, 0.77778475, 0.81708819, 0.85069839, 
                                      0.87921495, 0.90325159, 0.92339257, 0.94020581, 0.9541761, 0.96576543, 
                                      0.97529834, 0.98296667]) + 0.0027
        '''       
        # 32 levels in total

        self.phi_grid = np.linspace(-np.pi/2, np.pi/2, 180) #latitude grid
        self.lambda_grid = np.linspace(-np.pi, np.pi, 360) #longitude grid

        self.phi_c = np.deg2rad(40) #latitude of the center of the perturbation
        self.lambda_c = np.deg2rad(20) #longitude of the center of the perturbation

        # below we need to define some variables to better get the geopotential and the virtual temperature


        
    def zonal_u(self, eta, phi):
        if eta == 0:
            eta = 1e-10 # To avoid division by zero
        return -self.u0 * np.log(eta) * np.exp(-(np.log(eta)/self.b)**2) * (np.sin(2 * phi)**(2*self.n))

    def generate_zonal_u(self, IsPerturbation = False):
        zonal_u = np.zeros((32, 180, 360))
        for i in range(32):
            for j in range(180):
                for k in range(360):
                    zonal_u[i, j, k] = self.zonal_u(self.eta_level_accordingly[i], self.phi_grid[j])
        if IsPerturbation:
            r = np.zeros((32, 180, 360))
            for i in range(32):
                for j in range(180):
                    for k in range(360):
                        r[i, j, k] = self.a * np.arccos(np.sin(self.phi_c) * np.sin(self.phi_grid[j]) + np.cos(self.phi_c) * np.cos(self.phi_grid[j]) 
                                                        * np.cos(self.lambda_grid[k] - self.lambda_c))
            # correct the u_perturbation above
            u_perturbation = self.up * np.exp(-(r / self.R) ** 2)
            return np.expand_dims(zonal_u + u_perturbation, axis=0) 
        else:
            return np.expand_dims(zonal_u, axis=0)
        
    def generate_meridional_v(self):
        meridional_v = np.zeros((32, 180, 360))
        return np.expand_dims(meridional_v, axis=0)
    
    def generate_vertical_w(self):
        vertical_w = np.zeros((32, 180, 360))
        return np.expand_dims(vertical_w, axis=0)

    #f1, f2, f3, f4 are the functions of n and phi, these functions will return a (1, 32, 180, 360) grid
    def calculate_f1(self):
        F1_grid = np.zeros((1, 32, 180, 360))
        for i in range(32):  # Loop over eta levels (32 vertical levels)
            for j in range(180):  # Loop over phi grid (latitudes)
                phi = self.phi_grid[j]
                F1_sum = np.zeros(360)  # To store F1 for 360 longitudes
                for k in range(self.n + 1):  # Summation over k
                    binom_coeff = binom(self.n, k)  # Using binomial defined
                    term = binom_coeff * ((-1) ** k) / (2 * (k + self.n) + 1) * (np.cos(phi) ** (2 * (k + self.n) + 1))
                    F1_sum += term  # Summing the terms for each k
                F1_grid[0, i, j, :] = F1_sum  # Store in the grid for all longitudes
        return F1_grid
    
    def calculate_f2(self):
        F2_grid = np.zeros((1, 32, 180, 360))
        for i in range(32):  # Loop over eta levels (32 vertical levels)
            for j in range(180):  # Loop over phi grid (latitudes)
                phi = self.phi_grid[j]
                F2_sum = np.zeros(360)  # To store F2 for 360 longitudes
                for k in range(2 * self.n):  # Summation over k from 0 to 2n-1
                    binom_coeff = binom(2 * self.n - 1, k)  # Using binomial from scipy
                    term = binom_coeff * ((-1) ** k) / (2 * (k + 2 * self.n + 1)) * \
                           (np.sin(2 * phi) ** (2 * (k + 2 * self.n + 1)))
                    F2_sum += term  # Summing the terms for each k
                F2_grid[0, i, j, :] = F2_sum  # Store in the grid for all longitudes
        return F2_grid
    
    def calculate_f3(self):
        F3_grid = np.zeros((1, 32, 180, 360))
        sqrt_pi = np.sqrt(np.pi)
        for i in range(32):  # Loop over eta levels (32 vertical levels)
            for j in range(180):  # Loop over phi grid (latitudes)
                F3_sum = np.zeros(360)  # To store F3 for 360 longitudes
                for k in range(self.n + 1):  # Summation over k from 0 to n
                    binom_coeff = binom(self.n, k)  # Using binomial from scipy
                    gamma_ratio = gamma(k + self.n + 3/2) / gamma(k + self.n + 2)
                    term = binom_coeff * ((-1) ** k) / (2 * (k + self.n) + 1) * sqrt_pi * gamma_ratio
                    F3_sum += term  # Summing the terms for each k
                F3_grid[0, i, j, :] = F3_sum  # Store in the grid for all longitudes
        return F3_grid
    
    def calculate_f4(self):
        F4_grid = np.zeros((1, 32, 180, 360))
        for i in range(32):  # Loop over eta levels (32 vertical levels)
            for j in range(180):  # Loop over phi grid (latitudes)
                F4_sum = np.zeros(360)  # To store F4 for 360 longitudes
                for k in range(2 * self.n):  # Summation over k from 0 to 2n-1
                    binom_coeff = binom(2 * self.n - 1, k)  # Binomial coefficient
                    term = binom_coeff * ((-1) ** k) / (2 * (2 * self.n + k + 1)) * \
                           (2 / (2 * (2 * self.n + k + 1) + 1))
                    F4_sum += term  # Summing the terms for each k
                F4_grid[0, i, j, :] = F4_sum  # Store in the grid for all longitudes
        return F4_grid
    
    def calculate_u_eta(self):
        """Calculate u_eta and return a (1, 32, 180, 360) grid."""
        u_eta_grid = np.zeros((1, 32, 180, 360))  # Initialize a (1, 32, 180, 360) grid
        for i in range(32):  # Loop over all eta levels
            eta = self.eta_level_accordingly[i]
            if eta == 0:
                eta = 1e-10  # To avoid division by zero
            u_eta = self.u0 * np.log(eta) * np.exp(- (np.log(eta) / self.b) ** 2)  # Calculate u_eta for each eta level
            # Fill the entire (180, 360) grid with the same u_eta value for each eta level
            u_eta_grid[0, i, :, :] = u_eta  # Broadcast u_eta over the latitude and longitude grid
        return u_eta_grid

    def generate_geopotential(self):
        """Generate the geopotential height Φ(λ, φ, η) based on the provided formula."""
        # Retrieve u_eta, F1, F2, F3, and F4 grids
        u_eta = self.calculate_u_eta()  # u_eta (1, 32, 180, 360)
        F1 = self.calculate_f1()  # F1 (1, 32, 180, 360)
        F2 = self.calculate_f2()  # F2 (1, 32, 180, 360)
        F3 = self.calculate_f3()  # F3 (1, 32, 180, 360)
        F4 = self.calculate_f4()  # F4 (1, 32, 180, 360)
        
        # Initialize Φ grid
        geopotential = np.zeros((1, 32, 180, 360))
        
        # Compute the geopotential height
        for i in range(32):  # Loop over eta levels
            eta = self.eta_level_accordingly[i]
            first_term = (self.T_v0 * self.g / self.gamma) * (1 - eta ** (self.Rd * self.gamma / self.g))
            second_term = u_eta[0, i] * self.a * self.omega * (4 ** self.n) * (F3[0, i] - 2 * F1[0, i])
            third_term = (u_eta[0, i] ** 2) * (16 ** self.n) * (0.5 * F4[0, i] - F2[0, i])
            
            if eta < self.eta_t:
                first_term -= 4.8e5 * self.Rd * ((np.log(eta/self.eta_t) + 137/60) * self.eta_t ** 5 - 5 * self.eta_t ** 4 * eta +\
                                                 5 * self.eta_t ** 3 * eta ** 2 -10/3 * self.eta_t ** 2 * eta ** 3 +\
                                                      5/4 * self.eta_t * eta ** 4 - 1/5 * eta **5)
            
            geopotential[0, i] = first_term + second_term + third_term

            #print(self.T_v0, self.g, self.gamma, self.Rd, self.omega, eta)
        
        return geopotential

    def generate_phis(self):
        """Generate surface geopotential height Φ_s when η = 1 (1, 180, 360)."""
        # Calculate the u_eta, F1, F2, F3, and F4 for η = 1
        u_eta = self.u0 * np.log(1) * np.exp(- (np.log(1) / self.b) ** 2)  # u_eta when η = 1 is zero (log(1) = 0)
        
        F1 = self.calculate_f1()[0, -1, :, :]  # F1 at η = 1
        F2 = self.calculate_f2()[0, -1, :, :]  # F2 at η = 1
        F3 = self.calculate_f3()[0, -1, :, :]  # F3 at η = 1
        F4 = self.calculate_f4()[0, -1, :, :]  # F4 at η = 1
        
        # Since u_eta is zero, we only need to compute the terms involving u_eta^2
        phis = (u_eta ** 2) * (16 ** self.n) * (0.5 * F4 - F2)  # This simplifies the equation since u_eta = 0
        
        return np.expand_dims(phis, axis=0)

    def generate_DZ(self):
        """Generate DZ, the thickness of the layer (1, 32, 180, 360)."""
        geopotential_height = self.generate_geopotential()  # Get the geopotential height grid
        DZ = np.zeros((1, 32, 180, 360))  # Initialize DZ
        
        for i in range(31):  # Loop through 32 levels and calculate the difference between adjacent levels
            DZ[0, i] = (geopotential_height[0, i + 1] - geopotential_height[0, i]) / self.g
        
        # You can choose how to handle the topmost layer. Here I copy the value from the previous layer.
        DZ[0, 31] = DZ[0, 30]
        
        return DZ
    
    # potentional problem : on high latitude, the change of the geopotential height is significant

    def generate_T_v(self):
        virtual_temperature = np.zeros((1, 32, 180, 360))
        for i in range(32):
            eta = self.eta_level_accordingly[i]
            if eta == 0:
                eta = 1e-10 # To avoid division by zero
            first_term = self.T_v0 * eta ** (self.Rd * self.gamma / self.g)
            second_term = self.u0 / self.Rd * np.exp(- (np.log(eta) / self.b) ** 2) 
            third_term = 2 * (np.log(eta) / self.b) ** 2 - 1
            fourth_term = self.a * self.omega * (4 ** self.n) * (self.calculate_f3()[0, i] - 2 * self.calculate_f1()[0, i]) + \
                             (16 ** self.n) * self.calculate_u_eta()[0, i] * (self.calculate_f4()[0, i] - 2 * self.calculate_f2()[0, i])
            virtual_temperature[0, i] = first_term + second_term * third_term * fourth_term
            if eta < self.eta_t:
                virtual_temperature[0, i] += 4.8e5 * (self.eta_t - eta) ** 5

        return virtual_temperature
    
class basic_state_generator_gfdl(): #gfdl test
    def __init__(self, b = 2.0, n = 1, up = 1.0, gamma = 5e-3):
        self.u0 = 35.0 #reference zonal wind speed, define the zonal mean speed of the jet in troposphere
        self.b = b #nondimensional parameter representing the depth of the jet
        self.n = n #positive integer defining the width of the jet
        self.up = up #amplitude of the perturbation
        self.a = 6371.2e3  #earth radius
        self.g = 9.80616 #gravity acceleration
        self.Rd = 287.05 #gas constant for dry air
        self.omega = 7.2921e-5
        self.eta_t = 0.2 #tropopause level
        self.gamma = gamma #lapse rate
        self.R = self.a / 10 
        self.T_v0 = 288.0 #reference virtual temperature
        self.eta_level_accordingly = self.eta_level_accordingly = np.array([
                                                                0.00100, 0.00395, 0.00808, 0.01361, 0.02064,
                                                                0.02944, 0.04067, 0.05505, 0.07328, 0.09602,
                                                                0.12387, 0.15743, 0.19729, 0.24395, 0.29786,
                                                                0.35909, 0.42660, 0.49748, 0.56756, 0.63328,
                                                                0.69280, 0.74560, 0.79178, 0.83167, 0.86578,
                                                                0.89472, 0.91911, 0.93955, 0.95662, 0.97079,
                                                                0.98256, 0.99223
                                                            ], dtype=np.float32)
        
        '''
        np.array([
            0.00100, 0.00400, 0.00819, 0.01379, 0.02092, 0.02984, 0.04122, 0.05579,
            0.07420, 0.09705, 0.12497, 0.15855, 0.19840, 0.24503, 0.29889, 0.36004,
            0.42746, 0.49824, 0.56822, 0.63384, 0.69327, 0.74600, 0.79210, 0.83192,
            0.86598, 0.89487, 0.91923, 0.93964, 0.95667, 0.97083, 0.98257, 0.99223
        ])
        '''
        '''
        np.array([0.0, 0.00413128, 0.00966086, 0.01669672, 0.02549855, 0.03673121, 
                                      0.05111494, 0.06928385, 0.09183093, 0.11938442, 0.15252788, 0.19185386,
                                      0.23787391, 0.29103042, 0.35138178, 0.41791733, 0.48777655, 0.55683775, 
                                      0.62159457, 0.68024712, 0.73228272, 0.77778475, 0.81708819, 0.85069839, 
                                      0.87921495, 0.90325159, 0.92339257, 0.94020581, 0.9541761, 0.96576543, 
                                      0.97529834, 0.98296667]) + 0.0027
        '''       
        # 32 levels in total

        self.phi_grid = np.linspace(-np.pi/2, np.pi/2, 180) #latitude grid
        self.lambda_grid = np.linspace(-np.pi, np.pi, 360) #longitude grid

        self.eta0 = 0.252

        self.phi_c = np.deg2rad(30) #latitude of the center of the perturbation
        self.lambda_c = np.deg2rad(20) #longitude of the center of the perturbation

        # below we need to define some variables to better get the geopotential and the virtual temperature


        
    def zonal_u(self, eta, phi):
        if eta == 0:
            eta = 1e-10 # To avoid division by zero
        eta_v = np.pi / 2 * (eta - self.eta_t)
        return self.u0 * np.cos(eta_v) ** (3/2) * np.sin(2 * phi) ** (2 * self.n)
    
    def generate_zonal_u(self, IsPerturbation = False):
        zonal_u = np.zeros((32, 180, 360))
        for i in range(32):
            for j in range(180):
                for k in range(360):
                    zonal_u[i, j, k] = self.zonal_u(self.eta_level_accordingly[i], self.phi_grid[j])
        if IsPerturbation:
            r = np.zeros((32, 180, 360))
            for i in range(32):
                for j in range(180):
                    for k in range(360):
                        r[i, j, k] = self.a * np.arccos(np.sin(self.phi_c) * np.sin(self.phi_grid[j]) + np.cos(self.phi_c) * np.cos(self.phi_grid[j]) 
                                                        * np.cos(self.lambda_grid[k] - self.lambda_c))
            # correct the u_perturbation above
            u_perturbation = self.up * np.exp(-(r / self.R) ** 2)
            return np.expand_dims(zonal_u + u_perturbation, axis=0) 
        else:
            return np.expand_dims(zonal_u, axis=0)
        
    def generate_meridional_v(self):
        meridional_v = np.zeros((32, 180, 360))
        return np.expand_dims(meridional_v, axis=0)
    
    def generate_vertical_w(self):
        vertical_w = np.zeros((32, 180, 360))
        return np.expand_dims(vertical_w, axis=0)

    #f1, f2, f3, f4 are the functions of n and phi, these functions will return a (1, 32, 180, 360) grid
    def calculate_f1(self):
        F1_grid = np.zeros((1, 32, 180, 360))
        for i in range(32):  # Loop over eta levels (32 vertical levels)
            for j in range(180):  # Loop over phi grid (latitudes)
                phi = self.phi_grid[j]
                F1_sum = np.zeros(360)  # To store F1 for 360 longitudes
                for k in range(self.n + 1):  # Summation over k
                    binom_coeff = binom(self.n, k)  # Using binomial defined
                    term = binom_coeff * ((-1) ** k) / (2 * (k + self.n) + 1) * (np.cos(phi) ** (2 * (k + self.n) + 1))
                    F1_sum += term  # Summing the terms for each k
                F1_grid[0, i, j, :] = F1_sum  # Store in the grid for all longitudes
        return F1_grid
    
    def calculate_f2(self):
        F2_grid = np.zeros((1, 32, 180, 360))
        for i in range(32):  # Loop over eta levels (32 vertical levels)
            for j in range(180):  # Loop over phi grid (latitudes)
                phi = self.phi_grid[j]
                F2_sum = np.zeros(360)  # To store F2 for 360 longitudes
                for k in range(2 * self.n):  # Summation over k from 0 to 2n-1
                    binom_coeff = binom(2 * self.n - 1, k)  # Using binomial from scipy
                    term = binom_coeff * ((-1) ** k) / (2 * (k + 2 * self.n + 1)) * \
                           (np.sin(2 * phi) ** (2 * (k + 2 * self.n + 1)))
                    F2_sum += term  # Summing the terms for each k
                F2_grid[0, i, j, :] = F2_sum  # Store in the grid for all longitudes
        return F2_grid
    
    def calculate_f3(self):
        F3_grid = np.zeros((1, 32, 180, 360))
        sqrt_pi = np.sqrt(np.pi)
        for i in range(32):  # Loop over eta levels (32 vertical levels)
            for j in range(180):  # Loop over phi grid (latitudes)
                F3_sum = np.zeros(360)  # To store F3 for 360 longitudes
                for k in range(self.n + 1):  # Summation over k from 0 to n
                    binom_coeff = binom(self.n, k)  # Using binomial from scipy
                    gamma_ratio = gamma(k + self.n + 3/2) / gamma(k + self.n + 2)
                    term = binom_coeff * ((-1) ** k) / (2 * (k + self.n) + 1) * sqrt_pi * gamma_ratio
                    F3_sum += term  # Summing the terms for each k
                F3_grid[0, i, j, :] = F3_sum  # Store in the grid for all longitudes
        return F3_grid
    
    def calculate_f4(self):
        F4_grid = np.zeros((1, 32, 180, 360))
        for i in range(32):  # Loop over eta levels (32 vertical levels)
            for j in range(180):  # Loop over phi grid (latitudes)
                F4_sum = np.zeros(360)  # To store F4 for 360 longitudes
                for k in range(2 * self.n):  # Summation over k from 0 to 2n-1
                    binom_coeff = binom(2 * self.n - 1, k)  # Binomial coefficient
                    term = binom_coeff * ((-1) ** k) / (2 * (2 * self.n + k + 1)) * \
                           (2 / (2 * (2 * self.n + k + 1) + 1))
                    F4_sum += term  # Summing the terms for each k
                F4_grid[0, i, j, :] = F4_sum  # Store in the grid for all longitudes
        return F4_grid
    
    def calculate_u_eta(self):
        """Calculate u_eta and return a (1, 32, 180, 360) grid."""
        u_eta_grid = np.zeros((1, 32, 180, 360))  # Initialize a (1, 32, 180, 360) grid
        for i in range(32):  # Loop over all eta levels
            eta = self.eta_level_accordingly[i]
            if eta == 0:
                eta = 1e-10  # To avoid division by zero
            u_eta = self.u0 * np.log(eta) * np.exp(- (np.log(eta) / self.b) ** 2)  # Calculate u_eta for each eta level
            # Fill the entire (180, 360) grid with the same u_eta value for each eta level
            u_eta_grid[0, i, :, :] = u_eta  # Broadcast u_eta over the latitude and longitude grid
        return u_eta_grid

    def generate_geopotential(self):
        """Generate the geopotential height Φ(λ, φ, η) based on the provided formula."""
        
        # Initialize Φ grid
        geopotential = np.zeros((1, 32, 180, 360))
        
        # Compute the geopotential height
        for i in range(32):  # Loop over eta levels
            eta = self.eta_level_accordingly[i]
            eta_v = np.pi / 2 * (eta - self.eta0)
            first_term = (self.T_v0 * self.g / self.gamma) * (1 - eta ** (self.Rd * self.gamma / self.g))
            
            # Define φ (latitude) grid in radians for computation, assuming 180 latitudinal values from -90 to 90 degrees
            phi = np.linspace(-np.pi / 2, np.pi / 2, 180)
            
            # Compute second term according to the provided formula
            cos_eta_v = np.cos(eta_v) ** (3/2)
            u0_cos_eta_v = self.u0 * cos_eta_v
            
            inner_term1 = -2 * np.sin(phi) ** 6 * (np.cos(phi) ** 2 + 1 / 3) + 10 / 63
            inner_term2 = (8 / 5) * np.cos(phi) ** 3 * (np.sin(phi) ** 2 + 2 / 3) - np.pi / 4
            
            # Calculate the second term over the grid
            second_term = u0_cos_eta_v * (inner_term1 * u0_cos_eta_v + inner_term2 * self.a * self.omega)
            
            if eta < self.eta_t:
                first_term -= 4.8e5 * self.Rd * ((np.log(eta / self.eta_t) + 137 / 60) * self.eta_t ** 5 - 5 * self.eta_t ** 4 * eta +
                                                5 * self.eta_t ** 3 * eta ** 2 - (10 / 3) * self.eta_t ** 2 * eta ** 3 +
                                                (5 / 4) * self.eta_t * eta ** 4 - (1 / 5) * eta ** 5)
            
            # Combine first and second terms for final geopotential value
            geopotential[0, i] = first_term + second_term[:, np.newaxis]  # Broadcasting across longitude
            
        return geopotential


    def generate_phis(self):
        """Generate the surface geopotential height Φ(λ, φ) based on the provided formula at eta=eta0."""
        
        # Initialize Φ grid for the surface level
        surface_geopotential = np.zeros((180, 360))
        
        eta = 1  # Assuming eta0 represents the surface level
        eta_v = np.pi / 2 * (eta - self.eta0)
        first_term = (self.T_v0 * self.g / self.gamma) * (1 - eta ** (self.Rd * self.gamma / self.g))
        
        # Define φ (latitude) grid in radians for computation, assuming 180 latitudinal values from -90 to 90 degrees
        phi = np.linspace(-np.pi / 2, np.pi / 2, 180)
        
        # Compute second term according to the provided formula
        cos_eta_v = np.cos(eta_v) ** (3/2)
        u0_cos_eta_v = self.u0 * cos_eta_v
        
        inner_term1 = -2 * np.sin(phi) ** 6 * (np.cos(phi) ** 2 + 1 / 3) + 10 / 63
        inner_term2 = (8 / 5) * np.cos(phi) ** 3 * (np.sin(phi) ** 2 + 2 / 3) - np.pi / 4
        
        # Calculate the second term over the latitude grid
        second_term = u0_cos_eta_v * (inner_term1 * u0_cos_eta_v + inner_term2 * self.a * self.omega)
        
        # Combine first and second terms for final surface geopotential value
        surface_geopotential[:, :] = first_term + second_term[:, np.newaxis]  # Broadcasting across longitude
        
        return np.expand_dims(surface_geopotential, axis = 0)



    def generate_DZ(self):
        """Generate DZ, the thickness of the layer (1, 32, 180, 360)."""
        geopotential_height = self.generate_geopotential()  # Get the geopotential height grid
        DZ = np.zeros((1, 32, 180, 360))  # Initialize DZ
        
        for i in range(31):  # Loop through 32 levels and calculate the difference between adjacent levels
            DZ[0, i] = (geopotential_height[0, i + 1] - geopotential_height[0, i]) / self.g
        
        # You can choose how to handle the topmost layer. Here I copy the value from the previous layer.
        DZ[0, 31] = DZ[0, 30]
        
        return DZ
    
    # potentional problem : on high latitude, the change of the geopotential height is significant

    def generate_T_v(self):
        # Initialize Φ grid
        T_v = np.zeros((1, 32, 180, 360))
        
        # Compute the geopotential height
        for i in range(32):  # Loop over eta levels
            eta = self.eta_level_accordingly[i]
            eta_v = np.pi / 2 * (eta - self.eta0)
            first_term = self.T_v0 * eta ** (self.Rd * self.gamma / self.g)
            
            # Define φ (latitude) grid in radians for computation, assuming 180 latitudinal values from -90 to 90 degrees
            phi = np.linspace(-np.pi / 2, np.pi / 2, 180)
            
            # Compute second term according to the provided formula
            cos_eta_v = np.cos(eta_v) ** (1/2)
            u0_cos_eta_v = 3/4 * eta * np.pi * self.u0 / self.Rd * np.sin(eta_v) * cos_eta_v
            
            inner_term1 = -2 * np.sin(phi) ** 6 * (np.cos(phi) ** 2 + 1 / 3) + 10 / 63
            inner_term2 = (8 / 5) * np.cos(phi) ** 3 * (np.sin(phi) ** 2 + 2 / 3) - np.pi / 4
            
            # Calculate the second term over the grid
            second_term = u0_cos_eta_v * (inner_term1 * 2 * self.u0 * np.cos(eta_v) ** (3/2) + inner_term2 * self.a * self.omega)
            
            if eta < self.eta_t:
                first_term += 4.8e5 * (self.eta_t - eta) ** 5

            # Combine first and second terms for final geopotential value
            T_v[0, i] = first_term + second_term[:, np.newaxis]  # Broadcasting across longitude
            
        return T_v
    
import numpy as np
from scipy.integrate import cumtrapz, trapz
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class basic_state_generator_gfdl_shift:
    def __init__(self, b=2.0, n=1, up=1.0, gamma=5e-3, phi_shift=5, eta_fine_factor=1000):
        self.u0 = 35.0  # base u0
        self.b = b
        self.n = n
        self.up = up
        self.a = 6371.229e3
        self.g = 9.80616
        self.Rd = 287.05
        self.omega = 7.2921e-5
        self.eta_t = 0.2  # top layer
        self.eta_0 = 0.252
        self.gamma = gamma
        self.R = self.a / 10
        self.T_v0 = 288.0

        self.eta_level_original = np.array([
            0.00100, 0.00400, 0.00818, 0.01379, 0.02091,
            0.02983, 0.04121, 0.05579, 0.07420, 0.09704,
            0.12497, 0.15854, 0.19839, 0.24503, 0.29889,
            0.36004, 0.42745, 0.49824, 0.56822, 0.63383,
            0.69326, 0.74599, 0.79209, 0.83192, 0.86597,
            0.89487, 0.91922, 0.93963, 0.95667, 0.97082,
            0.98257, 0.99223, 1.00000
        ], dtype=np.float32)

        # vertical fine factoe
        self.eta_fine_factor = eta_fine_factor

        # original lat-lon grids
        self.phi_grid_original = np.linspace(-89.95, 89.95, 180 * 10) * np.pi / 180   # 1800
        self.lambda_grid = np.linspace(-179.95, 179.95, 360 * 10)  * np.pi / 180   # 3600

        self.phi_c = np.deg2rad(30 + phi_shift)
        self.lambda_c = np.deg2rad(20)

        self.phi_shift = np.deg2rad(phi_shift)

        # stored variables for fined geopotential
        self.Phi_prime_fine_2D = None  # (N_eta_fine, N_phi)
        self.Phi_prime_fine = None     # (N_eta_fine, N_phi, N_lambda)

    def zonal_u(self, eta, phi):
        """
        calculate the zonal wind on eta-lat grid (independent to lon)
        """
        eta = np.where(eta == 0, 1e-10, eta)
        eta_v = (np.pi / 2) * (eta[:, np.newaxis] - self.eta_0)  # (N_eta, 1)
        cos_eta_v = np.cos(eta_v) ** (3/2)  # (N_eta, 1)

        sin_term_north = np.sin(2 * (phi - self.phi_shift)) ** 2
        sin_term_south = np.sin(2 * (phi + self.phi_shift)) ** 2

        # add a dimension for boardcasting
        sin_term_north = sin_term_north[np.newaxis, :]  # (1, N_phi)
        sin_term_south = sin_term_south[np.newaxis, :]  # (1, N_phi)
        u = np.where(
            phi[np.newaxis, :] >= 0,
            self.u0 * cos_eta_v * sin_term_north,
            self.u0 * cos_eta_v * sin_term_south
        )

        # |phi| < self.phi_shift set to 0.
        mask = np.abs(phi) < self.phi_shift
        u[:, mask] = 0.0

        # |phi| > pi/2 + self.phi_shift set to 0.
        mask2 = np.abs(phi) > np.pi/2 + self.phi_shift
        u[:, mask2] = 0.0

        return u  # (N_eta, N_phi)

    def generate_zonal_u_stored(self, eta_grid, phi_grid, IsPerturbation=False):
        """
        generate wind field.
        """

        base_u_2D = self.zonal_u(eta_grid, phi_grid)  # (N_eta, N_phi)

        if not IsPerturbation:
            #print(base_u_2D.shape)
            #print(base_u_2D[])
            return base_u_2D  # (N_eta, N_phi)
        else:
            u_3D = np.repeat(base_u_2D[:, :, np.newaxis], len(self.lambda_grid), axis=2)
         
            phi_2D = phi_grid[:, np.newaxis]  # (N_phi, 1)
            lambd_1D = self.lambda_grid[np.newaxis, :]  # (1, N_lambda)

           
            argument = (
                np.sin(self.phi_c) * np.sin(phi_2D) +
                np.cos(self.phi_c) * np.cos(phi_2D) * np.cos(lambd_1D - self.lambda_c)
            )
            argument = np.clip(argument, -1.0, 1.0)
            r = self.a * np.arccos(argument)  # (N_phi, N_lambda)
            u_perturbation_2D = self.up * np.exp(-(r / self.R) ** 2)

            # broadcast到 eta 维度: (N_eta, N_phi, N_lambda)
            u_perturbation_3D = u_perturbation_2D[np.newaxis, :, :]
            u_3D += u_perturbation_3D

            return u_3D

    def generate_zonal_u(self, IsPerturbation=False):
        """
        return 4-D zonal wind
        """
        eta = (self.eta_level_original[1:] + self.eta_level_original[:-1]) * 0.5
        u_stored = self.generate_zonal_u_stored(
            eta,
            self.phi_grid_original,
            IsPerturbation=IsPerturbation
        )
        
        if u_stored.ndim == 2:
            # (N_eta, N_phi) -> (N_eta, N_phi, N_lambda)
            u_3D = np.repeat(u_stored[:, :, np.newaxis], len(self.lambda_grid), axis=2)
        else:
            u_3D = u_stored
        
        #print(u_3D.shape)
        #u_3D = u_3D[:-1,:,:] #(33->32)
        return np.expand_dims(u_3D, axis=0)

    def generate_meridional_v(self):
        """
        return 4-D meridinal wind
        """
        meridional_v = np.zeros((32, 180 * 10, 360 * 10))
        return np.expand_dims(meridional_v, axis=0)

    def generate_vertical_w(self):
        """
        return 4-D vertical motion
        """
        vertical_w = np.zeros((32, 180 * 10, 360 * 10))
        return np.expand_dims(vertical_w, axis=0)

    def generate_geopotential_prime_fine(self, plot_debug=False):
        """
        numerical method to generate fined geopotential
        """
        N_eta_original = len(self.eta_level_original)
        eta_fine_factor = self.eta_fine_factor
        N_eta_fine = N_eta_original * eta_fine_factor
        eta_grid_fine = np.linspace(
            self.eta_level_original[0],
            self.eta_level_original[-1],
            N_eta_fine
        )

        # ---- (1) u_fine(eta,phi) ----
        u_fine_2D = self.generate_zonal_u_stored(
            eta_grid_fine,
            self.phi_grid_original,
            IsPerturbation=False
        )  # (N_eta_fine, N_phi)

        a = self.a
        Omega = self.omega

        sin_phi_1D = np.sin(self.phi_grid_original)    # (N_phi,)
        tan_phi_1D = np.tan(self.phi_grid_original)    # (N_phi,)
        # avoid infinity at poler region
        tan_phi_1D = np.where(
            np.abs(self.phi_grid_original) >= (np.pi/2 - 1e-6),
            0.0,
            tan_phi_1D
        )

        sin_phi_2D = sin_phi_1D[np.newaxis, :]  # (1, N_phi)
        tan_phi_2D = tan_phi_1D[np.newaxis, :]  # (1, N_phi)

        # ---- (2) caculate integrand f_2D, shape: (N_eta_fine, N_phi) ----
        f_2D = -a * u_fine_2D * (2 * Omega * sin_phi_2D + (u_fine_2D / a) * tan_phi_2D)

        # ---- (3) integrate along phi cumtrapz ----
        Phi_prime_fine_2D = cumtrapz(
            f_2D,
            self.phi_grid_original,
            axis=1,
            initial=0.0
        )  # (N_eta_fine, N_phi)

        # ---- (4) minus the average to keep the whole phi' to 0 ----
        cos_phi_1D = np.cos(self.phi_grid_original)
        Phi_cos_2D = Phi_prime_fine_2D * cos_phi_1D[np.newaxis, :]  # (N_eta_fine, N_phi)
        integral_1D = trapz(Phi_cos_2D, self.phi_grid_original, axis=1)  # (N_eta_fine,)
        Phi_mean_1D = integral_1D / 2.0  # (N_eta_fine,)


        Phi_mean_2D = Phi_mean_1D[:, np.newaxis]  # (N_eta_fine, 1)
        Phi_fine_adjusted_2D = Phi_prime_fine_2D - Phi_mean_2D

        # ---- (5) add first_term ----
        first_term = (self.T_v0 * self.g / self.gamma) * (
            1 - eta_grid_fine ** (self.Rd * self.gamma / self.g)
        )  # (N_eta_fine,)

        # if eta < eta_t, add some modification
        mask = eta_grid_fine < self.eta_t

        first_term[mask] -= 4.8e5 * self.Rd * (
            (np.log(eta_grid_fine[mask] / self.eta_t) + 137/60) * self.eta_t**5
            - 5 * self.eta_t**4 * eta_grid_fine[mask]
            + 5 * self.eta_t**3 * eta_grid_fine[mask]**2
            - (10/3) * self.eta_t**2 * eta_grid_fine[mask]**3
            + (5/4) * self.eta_t * eta_grid_fine[mask]**4
            - (1/5) * eta_grid_fine[mask]**5
        )


        first_term_2D = first_term[:, np.newaxis]
        Phi_prime_fine_2D = Phi_fine_adjusted_2D + first_term_2D
        '''

        Phi_prime_fine_3D = np.repeat(
            Phi_prime_fine_2D[:, :, np.newaxis],
            len(self.lambda_grid),
            axis=2
        )
        It seems that 3D results is resource commending and useless.
        '''
        # save 2-D results
        self.Phi_prime_fine_2D = Phi_prime_fine_2D
        #self.Phi_prime_fine = Phi_prime_fine_3D

        if plot_debug:
            plt.figure(figsize=(12, 5))
            plt.plot(np.degrees(self.phi_grid_original), f_2D[0, :], label='Integrand f(phi) for first eta')
            plt.title('Integrand f(phi) for first eta')
            plt.xlabel('phi (degrees)')
            plt.ylabel('f(phi)')
            plt.legend()
            plt.grid(True)
            plt.show()

            plt.figure(figsize=(12, 5))
            plt.plot(np.degrees(self.phi_grid_original), Phi_prime_fine_2D[0, :],
                     label="Phi_prime_fine_2D (Geopotential) for first eta")
            plt.title('Phi_prime_fine_2D for first eta')
            plt.xlabel('phi (degrees)')
            plt.ylabel("Phi'_fine (m^2/s^2)")
            plt.legend()
            plt.grid(True)
            plt.show()

    def generate_geopotential(self, plot_debug=False):
        """
        interpolate to eta_level_original
        """
        if self.Phi_prime_fine_2D is None:
            self.generate_geopotential_prime_fine(plot_debug=plot_debug)

        N_eta_fine = self.Phi_prime_fine_2D.shape[0]
        eta_grid_fine = np.linspace(self.eta_level_original[0], self.eta_level_original[-1], N_eta_fine)


        Phi_prime_fine_2D = self.Phi_prime_fine_2D

        interp_func = interp1d(
            eta_grid_fine,
            Phi_prime_fine_2D,  
            kind='cubic',
            axis=0, 
            bounds_error=False,
            fill_value="extrapolate"
        )
        # (N_eta_original, N_phi)
        Phi_original_2D = interp_func(self.eta_level_original)

        # (N_eta_original, N_phi) -> (N_eta_original, N_phi, N_lambda)
        Phi_original_3D = np.repeat(
            Phi_original_2D[:, :, np.newaxis],
            len(self.lambda_grid),
            axis=2
        )
        # (1, N_eta_original, N_phi, N_lambda)
        Phi = np.expand_dims(Phi_original_3D, axis=0)
        return Phi

    def generate_T_v(self):
        """
        calculate T_v
        """
        if self.Phi_prime_fine_2D is None:
            self.generate_geopotential_prime_fine()

        N_eta_fine = self.Phi_prime_fine_2D.shape[0]
        eta_grid_fine = np.linspace(self.eta_level_original[0], self.eta_level_original[-1], N_eta_fine)

        #  dPhi'/deta
        Phi_prime_fine_2D = self.Phi_prime_fine_2D  # (N_eta_fine, N_phi)
        delta_eta = eta_grid_fine[1] - eta_grid_fine[0]

        # calculate vertical difference
        dPhi_prime_deta_2D = np.zeros_like(Phi_prime_fine_2D)  # (N_eta_fine, N_phi)

        dPhi_prime_deta_2D[1:-1] = (
            Phi_prime_fine_2D[2:, :] - Phi_prime_fine_2D[:-2, :]
        ) / (2 * delta_eta)

        dPhi_prime_deta_2D[0] = (
            Phi_prime_fine_2D[1, :] - Phi_prime_fine_2D[0, :]
        ) / delta_eta

        dPhi_prime_deta_2D[-1] = (
            Phi_prime_fine_2D[-1, :] - Phi_prime_fine_2D[-2, :]
        ) / delta_eta

        #  T_v_fine_2D = -(eta / Rd) * dPhi'/deta
        eta_2D = eta_grid_fine[:, np.newaxis]  # (N_eta_fine, 1)
        T_v_fine_2D = - (eta_2D / self.Rd) * dPhi_prime_deta_2D  # (N_eta_fine, N_phi)

        # interpolate
        interp_func = interp1d(
            eta_grid_fine,
            T_v_fine_2D,
            kind='cubic',
            axis=0,
            bounds_error=False,
            fill_value="extrapolate"
        )
        eta = (self.eta_level_original[1:] + self.eta_level_original[:-1]) * 0.5
        T_v_original_2D = interp_func(eta)  # (N_eta, N_phi)

        # (N_eta, N_phi, N_lambda)
        T_v_3D = np.repeat(
            T_v_original_2D[:, :, np.newaxis],
            len(self.lambda_grid),
            axis=2
        )
        #  (1, N_eta, N_phi, N_lambda)
        T_v = np.expand_dims(T_v_3D, axis=0)
        return T_v#[:,:-1,:,:]

    def generate_DZ(self):
        """
        DZ = (Phi[k+1] - Phi[k]) / g with shape: (1, N_eta, N_phi, N_lambda)。
        """
        Phi = self.generate_geopotential()  # (1, N_eta, N_phi, N_lambda)
        DZ = np.zeros_like(Phi)
        DZ[:, :-1, :, :] = (Phi[:, 1:, :, :] - Phi[:, :-1, :, :]) / self.g
        DZ[:, -1, :, :] = DZ[:, -2, :, :]
        return DZ[:,:-1,:,:]
    
    def generate_phis(self):
        Phi = self.generate_geopotential()
        return Phi[:,-1,:,:]




if __name__ == '__main__':
    u = basic_state_generator_gfdl_shift()
    T_v = u.generate_T_v()
    print("T_v shape =", T_v.shape)