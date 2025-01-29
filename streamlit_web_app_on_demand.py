import streamlit as st
import numpy as np
import pandas as pd
# from st_aggrid import AgGrid, GridOptionsBuilder
from streamlit_echarts import st_echarts
import datetime
import json
import itertools
import time
import math
import os
from os import listdir
from os.path import isfile, join
import itertools
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from streamlit.components.v1 import html
import webbrowser
import fnmatch
import pvlib
import io
import requests
import plotly.graph_objects as go

# Setup the session_state parameters first, this should include things like...
#  Prop details - annual elec/gas, postcode, lat-long, orientation, no. bedrooms etc
#  my technology upgrades
# This is so that we can persist these and have them available in the dialog boxes

# We should set some default parameters, for usage, location, and upgrades
# https://www.ofgem.gov.uk/average-gas-and-electricity-usage

# We should also have a "so-what" page - what can you do next?
# Find grants, get a loan (lendology, egg), get insulated (furbnow?), get a retrofit survey, find an installer


st.set_page_config(layout="wide")
# st.session_state.postcode = None

def set_results_require_rerun():
	st.session_state.results_present = False

@st.dialog("Property Details", width="large")
def set_property_details():

	c1, c2 = st.columns([1,3])

	with c1:

		if 'postcode' in st.session_state:
			postcode = st.text_input("Postcode", st.session_state.postcode, help='To calculate the solar potential, outside air temperature, & match to available grants/installers')
		else:
			postcode = st.text_input("Postcode", 'MK5 8HD', help='To calculate the solar potential, outside air temperature, & match to available grants/installers')


		property_type = st.selectbox(
	    "Home Type",
	    ("Flat/1 Bed House","2-3 Bed House","4+ Bed House","Custom"),
	    index=2
	    )

		if property_type != 'Custom':

			if property_type == 'Flat/1 Bed House':
				elec_val = 1800
				gas_val = 7500

			if property_type == '2-3 Bed House':
				elec_val = 2700
				gas_val = 11500
			
			if property_type == '4+ Bed House':
				elec_val = 4100
				gas_val = 17000			

			annual_electricity_consumption_kWh = st.number_input('Annual Elec Usage (kWh)', 
				min_value=100, max_value=25000, value=elec_val, step=1,
				help='Excluding demand from EV, Heat Pumps, Batteries, Solar PV - Defaults to UK avg',
				disabled = True,
				on_change=set_results_require_rerun)

			annual_user_gas_demand_kWh = st.number_input('Annual Gas Usage (kWh)', 
														min_value=0, max_value=50000, 
														value=gas_val, step=1,
														help='Defaults to UK avg',
														disabled = True,
														on_change=set_results_require_rerun)

		else:
			annual_electricity_consumption_kWh = st.number_input('Annual Elec Usage (kWh)', 
			min_value=100, max_value=25000, value=2700, step=1,
			help='Excluding demand from EV, Heat Pumps, Batteries, Solar PV - Defaults to UK avg',
			disabled = False,
			on_change=set_results_require_rerun)

			annual_user_gas_demand_kWh = st.number_input('Annual Gas Usage (kWh)', 
													min_value=0, max_value=50000, 
													value=11500, step=1,
													help='Defaults to UK avg',
													disabled = False,
													on_change=set_results_require_rerun)

	with c2:
		if postcode != '':
			response = requests.get("https://api.postcodes.io/postcodes/"+postcode)
			if response.status_code == 200:
				postcode_api_results = response.json()
				latitude = postcode_api_results['result']['latitude']
				longitude = postcode_api_results['result']['longitude']
			
				st.map(data=pd.DataFrame(data={'latitude':[latitude],
												'longitude':[longitude]}
												),
						size=10., zoom=15, height=300
				)

				st.session_state.property_details_set = True
				st.session_state.latitude = latitude
				st.session_state.longitude = longitude
				st.session_state.annual_electricity_consumption_kWh = annual_electricity_consumption_kWh
				st.session_state.annual_user_gas_demand_kWh = annual_user_gas_demand_kWh


				if st.button("Confirm Details"):
					st.session_state.postcode = postcode.upper()
					st.rerun()

			else:
				st.write('Please Input A Valid UK Postcode')			
		else:
			st.write('Please Input A Valid UK Postcode')			
		

@st.dialog("Upgrades", width="large")
def set_upgrades(vehicles_df, solar_pv_systems_df, heating_systems_df,
				 battery_storage_systems_df, ev_chargers_df, energy_tariffs_df,
				 locations_df , installers_df):




	technology_options = []

	tab1, tab2, tab3, tab4, tab5 = st.tabs(["Heating", "Battery","Solar PV","Tariff","Vehicle"])


	# if the technology choices has been selected, we need to put that as the selected option...
	with tab1:


		if 'technology_choices' in st.session_state:
			current_heating_idx = list(heating_systems_df['heating_system_name'].unique()).index(st.session_state.technology_choices['current_heating_system'])
			future_heating_idx = list(heating_systems_df['heating_system_name'].unique()).index(st.session_state.technology_choices['future_heating_system'])

		else:
			current_heating_idx = 0
			future_heating_idx = 1

		current_heating_system = st.selectbox('Current',
				options = heating_systems_df['heating_system_name'].unique(),
				index = current_heating_idx,
				on_change=set_results_require_rerun
				)
		st.markdown("""---""")

		future_heating_system = st.selectbox(label='Future',
		options=heating_systems_df['heating_system_name'].unique(),
		index=future_heating_idx,
		disabled=False,
		on_change=set_results_require_rerun
		)
		if future_heating_system != current_heating_system:
			technology_options.append('Heating')
			
	with tab2:


		if 'technology_choices' in st.session_state:
			current_battery_storage_idx = list(battery_storage_systems_df['battery_storage_name'].unique()).index(st.session_state.technology_choices['current_battery_storage_system'])
			future_battery_storage_idx = list(battery_storage_systems_df['battery_storage_name'].unique()).index(st.session_state.technology_choices['battery_storage_option'])
			current_battery_num_units = st.session_state.technology_choices['current_battery_num_units']

		else:
			current_battery_storage_idx = 0
			future_battery_storage_idx = 1
			current_battery_num_units = 0


		current_battery_storage_system = st.selectbox('Current',
				 options=battery_storage_systems_df['battery_storage_name'].unique(),
				 index = current_battery_storage_idx,
				 on_change=set_results_require_rerun
				 )
			 
		if current_battery_storage_system == 'No Battery Storage':
			current_battery_num_units = st.slider('Current Battery Num Units', 1, 5, 
										0, step=1,
										help='Defaults to 1',
										disabled=True)

		else:
			current_battery_num_units = st.slider('Current Battery Num Units', 1, 5, 1, step=1,
										help='Defaults to 1',
										on_change=set_results_require_rerun)

		st.markdown("""---""")

		battery_storage_option = st.selectbox(
			label = 'Future',
			options = battery_storage_systems_df['battery_storage_name'].unique(),
			index = future_battery_storage_idx,
			disabled=False,
			on_change=set_results_require_rerun
		)

		if battery_storage_option != 'No Battery Storage':
			battery_number_units = st.slider('Number of Battery Units', 1, 6, 1, step=1, 
												help='Defaults to 1 unit', 
												on_change=set_results_require_rerun)
		else:
			battery_number_units = st.slider('Future Battery Num Units', 0, 5, 0, step=1,
										help='Set to zero',
										disabled=True)
			battery_number_units = 0
		
		
		if battery_storage_option != current_battery_storage_system:
			technology_options.append('Battery')

		# battery_storage_systems_df = battery_storage_systems_df.loc[battery_storage_systems_df['battery_storage_name'].isin([current_battery_storage_system]+[battery_storage_option])]			

	with tab3:

		if 'technology_choices' in st.session_state:
			current_solar_pv_system_idx = list(solar_pv_systems_df['solar_pv_name'].unique()).index(st.session_state.technology_choices['current_solar_pv_system'])
			future_solar_pv_system_idx = list(solar_pv_systems_df['solar_pv_name'].unique()).index(st.session_state.technology_choices['solar_pv_option'])
			current_solar_PV_Wp = st.session_state.technology_choices['current_solar_PV_Wp']
			future_solar_PV_Wp = st.session_state.technology_choices['future_solar_pv_power_Wp']
			azimuth = st.session_state.technology_choices['azimuth']
			tilt = st.session_state.technology_choices['tilt']

		else:
			current_solar_pv_system_idx = 0
			future_solar_pv_system_idx = 1
			current_solar_PV_Wp = 0
			future_solar_PV_Wp = 4000
			azimuth = 180
			tilt = 35
			

		current_solar_pv_system = st.selectbox(
			 'Current',
			 solar_pv_systems_df['solar_pv_name'].unique(),
			 index = current_solar_pv_system_idx,
			 help = 'Assumes system facing due-south, 35degs slope',
			 on_change=set_results_require_rerun
			 )

		if current_solar_pv_system != 'No Solar PV':					
			current_solar_PV_Wp = st.slider('Current Solar PV size (Wp)', 0, 10000, 
									current_solar_PV_Wp, 
									step=100,
									help='Defaults to 4kWp', 
									on_change=set_results_require_rerun)
		else:
			current_solar_PV_Wp = 0

		st.markdown("""---""")
		solar_pv_option = st.selectbox(
						label = 'Future',
						options = solar_pv_systems_df['solar_pv_name'].unique(),
						index = future_solar_pv_system_idx,
						disabled=False,
						help = 'Assumes system facing due-south, 35degs slope',
						on_change=set_results_require_rerun
						)

		if solar_pv_option != 'No Solar PV':
			future_solar_PV_Wp = st.slider('Future Solar PV size (Wp)',0,10000,
											future_solar_PV_Wp,step=100, help='Defaults to 4kWp',
											on_change=set_results_require_rerun)
		else:
			future_solar_PV_Wp = st.slider('Future Solar PV size (Wp)',0,10000,
											0,step=100, help='Defaults to 0',
											on_change=set_results_require_rerun,
											disabled=True)

		if (current_solar_pv_system != 'No Solar PV') | (solar_pv_option != 'No Solar PV'):
			azimuth = st.number_input("Azimuth", value=azimuth, min_value=0, max_value=359, 
										on_change=set_results_require_rerun,
										help='Azimuth of the PV system - 180 is due south, 90 is due east, 270 is due west')
			tilt = st.number_input("Tilt", value=tilt, min_value=0, max_value=90, 
									on_change=set_results_require_rerun,
									help='Tilt of the PV system - 0 is flat, 90 is perpendicular to roof')
		else:
			azimuth = st.number_input("Azimuth", value=azimuth, min_value=0, max_value=359, 
										on_change=set_results_require_rerun,
										help='Azimuth of the PV system - 180 is due south, 90 is due east, 270 is due west',
										disabled=True)
			tilt = st.number_input("Tilt", value=tilt, min_value=0, max_value=90, 
									on_change=set_results_require_rerun,
									help='Tilt of the PV system - 0 is flat, 90 is perpendicular to roof',
									disabled=True)
		
		if solar_pv_option != current_solar_pv_system:
			technology_options.append('Solar PV')

		future_solar_pv_power_Wp = future_solar_PV_Wp

	with tab4:

		if 'technology_choices' in st.session_state:
			current_energy_tariff_idx = list(energy_tariffs_df['tariff_name'].unique()).index(st.session_state.technology_choices['current_energy_tariff'])
			future_energy_tariff_idx = list(energy_tariffs_df['tariff_name'].unique()).index(st.session_state.technology_choices['energy_tariff_option'])
			export_limit_kW = st.session_state.technology_choices['export_limit_kW']
		else:
			current_energy_tariff_idx = 0
			future_energy_tariff_idx = 3
			export_limit_kW = 3.68

		export_limit_kW = st.number_input('Export Limit (kW)', value=export_limit_kW, 
										   min_value=3.68, step = 0.01,
										   help='- DNOs (Distribution Network Operators) typically limit export to 3.68kW (G98).  Can be extended to 50kW with a successful G99 application',
										   on_change=set_results_require_rerun)
		st.markdown("""---""")		
		current_energy_tariff = st.selectbox('Current',
				 energy_tariffs_df['tariff_name'].unique(),
				 index = current_energy_tariff_idx,
				 on_change=set_results_require_rerun
				 )

		st.markdown("""---""")

		energy_tariff_option = st.selectbox(
		label = 'Future',
		options = energy_tariffs_df['tariff_name'].unique(),
		index = future_energy_tariff_idx,
		disabled=False,
		on_change=set_results_require_rerun
		)

		if  energy_tariff_option != current_energy_tariff:
			technology_options.append('Tariff')
				
	with tab5:
		annual_miles_driven = st.number_input('Annual Miles Driven', min_value=0, 
												max_value=100000, value=10000, step=100,
												help='Defaults to UK avg',
												on_change=set_results_require_rerun)
	
	
		home_departure_time, home_arrival_time = st.slider('Home Departure and Arrival Time', 
						value=(datetime.time(7, 0), datetime.time(18, 0)),
						step=datetime.timedelta(minutes=30),
						on_change=set_results_require_rerun)

		home_departure_hh_period = int((2*home_departure_time.hour)+(home_departure_time.minute/30.))
		home_arrival_hh_period = int((2*home_arrival_time.hour)+(home_arrival_time.minute/30.))
		arrival_departure_delta_n_hh_periods = home_departure_hh_period - home_arrival_hh_period + 48
	
		vehicle_fuel_cost_per_litre = st.slider('Vehicle Fuel Cost (£/litre)', 1.00, 2.50, 1.60, step=0.01)
		current_vehicle = st.selectbox('Current Car',
				 vehicles_df['vehicle_name'].unique(),
				 on_change=set_results_require_rerun
				 )
		st.markdown("""---""")			

		future_vehicle = st.selectbox(
		label = 'Future Car',
		options = vehicles_df['vehicle_name'].unique(),
		index = 1,
		disabled=False,
		on_change=set_results_require_rerun
		)

		if future_vehicle != current_vehicle:
			technology_options.append('Vehicle')				

	# Can we have 1 dictionary that encompasses all these parameters?
	if st.button("Confirm Details"):
		st.session_state.n_technology_options = len(technology_options)
		st.session_state.technology_choices = {
			'current_battery_storage_system':current_battery_storage_system,
			'current_battery_num_units':current_battery_num_units,
			'battery_storage_option':battery_storage_option,
			'battery_number_units':battery_number_units,
			'current_heating_system':current_heating_system,
			'future_heating_system':future_heating_system,
			'current_solar_PV_Wp':current_solar_PV_Wp,
			'current_solar_pv_system':current_solar_pv_system,
			'future_solar_pv_power_Wp':future_solar_pv_power_Wp,
			'solar_pv_option':solar_pv_option,
			'azimuth':azimuth,
			'tilt':tilt,
			'current_energy_tariff':current_energy_tariff,
			'energy_tariff_option':energy_tariff_option,
			'current_vehicle':current_vehicle,
			'future_vehicle':future_vehicle,
			'home_departure_hh_period':home_departure_hh_period,
			'home_arrival_hh_period':home_arrival_hh_period,
			'arrival_departure_delta_n_hh_periods':arrival_departure_delta_n_hh_periods,
			'annual_miles_driven':annual_miles_driven,
			'export_limit_kW':export_limit_kW,
			'vehicle_fuel_cost_per_litre':vehicle_fuel_cost_per_litre
		}		
		st.rerun()

	
	return heating_systems_df, battery_storage_systems_df, solar_pv_systems_df, energy_tariffs_df, vehicles_df

@st.dialog("Installers Covering Your Area:", width="large")
def show_next_steps(installers_df, postcode_region):
	

	suitable_installer_ids = []
	for n in range(len(installers_df.index)):
		installer_regions = installers_df['region'].values[n]
		if postcode_region in installer_regions:
			suitable_installer_ids.append(installers_df['installer_id'].values[n])

	suitable_intallers_df = installers_df.loc[installers_df['installer_id'].isin(suitable_installer_ids)]
	suitable_intallers_df.rename(columns={'name':'Company','products':'Products Offered'},inplace=True)
	display_cols = ['Company','url','Products Offered']
	st.markdown(
	                """
	                <style>
	                [data-testid="stElementToolbar"] {
	                    display: none;
	                }
	                </style>
	                """,
	                unsafe_allow_html=True
	            )
	st.dataframe(suitable_intallers_df[display_cols], hide_index=True,
				column_config={
        			"url": st.column_config.LinkColumn(
            		"Link", display_text="Visit Installer"
        			)}
        		)

	st.subheader('You may also be eligible for the following loans & grants:')
	st.write("- [Lendology](https://www.lendology.org.uk/) - Home improvement loans for homeowners, funded by local councils")
	st.write("- [ECO4](https://eco4.org.uk/) - If you own/rent a property that is heated with electricity and receive income-related benefits")
	st.write("- [Boiler Upgrade Scheme](https://www.gov.uk/apply-boiler-upgrade-scheme) - a grant to cover part of the cost of replacing fossil fuel heating systems with a heat pump or biomass boiler.")


@st.dialog("Details", width="large")
def show_details():
	st.write('YPlaceholder - detailed analysis')

## Load in all open-source data files  - these do not depend on user selections
@st.cache_data
def load_common_inputs():
	## Load in all open-source data files ##

	# Loading in Vehicles data JSON
	with open('vehicles.json') as json_file:
		data = json.load(json_file)
	vehicles_df = pd.DataFrame(data).fillna(0).drop(labels=['objectType','country'],axis=1)

	# Loading in Solar PV System data JSON
	with open('solarpv_systems.json') as json_file:
		data = json.load(json_file)

	solar_pv_systems_df = pd.DataFrame(data).fillna(0).drop(labels=['objectType','country'],axis=1)
# 	solar_pv_systems_df

	# Loading in Heating Systems data JSON
	with open('heating_systems.json') as json_file:
		data = json.load(json_file)

	heating_systems_df = pd.DataFrame(data).fillna(0).drop(labels=['objectType','country'],axis=1)
# 	heating_systems_df

	# Loading in Battery Storage Systems data JSON
	with open('battery_storage_systems.json') as json_file:
		data = json.load(json_file)

	battery_storage_systems_df = pd.DataFrame(data).fillna(0).drop(labels=['objectType','country'],axis=1)

	# Loading in EV Home Charger data JSON
	with open('ev_chargers.json') as json_file:
		data = json.load(json_file)
	ev_chargers_df = pd.DataFrame(data).fillna(0).drop(labels=['objectType','country'],axis=1)

	# Loading in Energy Tariffs (Electricity & Gas) data JSON
	with open('energy_tariffs.json') as json_file:
		data = json.load(json_file)
	energy_tariffs_df = pd.DataFrame(data).fillna(0)

	with open('locations.json') as json_file:
		data = json.load(json_file)
	locations_df = pd.DataFrame(data).fillna(0)	

	with open('installers.json') as json_file:
		data = json.load(json_file)
	installers_df = pd.DataFrame(data).fillna(0)	

	postcodes_df = pd.read_csv('postcodes.csv')
	
	return vehicles_df, solar_pv_systems_df, heating_systems_df, battery_storage_systems_df, \
			ev_chargers_df, energy_tariffs_df, locations_df, installers_df, postcodes_df


def generate_half_hourly_electricity_baseload(profile_name, annual_electricity_consumption_kWh, home_arrival_hh_period):

	# Energy consumption profile
	infile = profile_name+".csv"  # This should be selectable by customers from a pre-loaded set (class 1, working away from home, WFH)

	# Creating a normalised demand profile from Elexon Class 1 data, using 2021 as a base year
	# Downloaded from https://ukerc.rl.ac.uk/DC/cgi-bin/edc_search.pl/?WantComp=42

	baseload_profile_df = pd.read_csv(infile)

	demand_cols = [x for x in baseload_profile_df.columns.values if x!='Time' ]
	baseload_profile_df['annual_avg_demand_kW'] = baseload_profile_df[demand_cols].mean(axis=1)


	baseload_profile_df['electricity_demand_normalised'] = baseload_profile_df['annual_avg_demand_kW'] / baseload_profile_df['annual_avg_demand_kW'].sum()
	baseload_profile_df[['hour', 'minute']] = baseload_profile_df['Time'].str.split(pat=':', n=-1, expand=True).astype(int)

	baseload_profile_df['profile_id'] = 0
	baseload_profile_df['profile_name'] = 'Elexon Class 1'

	keep_cols = ['profile_id','profile_name','hour','minute','electricity_demand_normalised']
	drop_cols = [x for x in baseload_profile_df.columns.values if x not in keep_cols]

	baseload_profile_df.drop(labels=drop_cols, inplace=True, axis=1)

	hh = pd.date_range("2019-01-01T00:00:00", "2019-12-31T23:30:00", freq="30min")
	half_hourly_df = pd.DataFrame(data={'datetime':pd.date_range("2019-01-01T00:00:00", 
																 "2019-12-31T23:30:00", 
																 freq="30min")}
								 )

	ds = half_hourly_df['datetime'].values

	half_hourly_df['day_of_year'] = half_hourly_df['datetime'].dt.dayofyear
	half_hourly_df['month'] = half_hourly_df['datetime'].dt.month
	half_hourly_df['hour'] = half_hourly_df['datetime'].dt.hour
	half_hourly_df['minute'] = half_hourly_df['datetime'].dt.minute

	half_hourly_df = pd.merge(half_hourly_df, baseload_profile_df,
							  on=['hour','minute'], how='left')

	profile_grouped_df = half_hourly_df.groupby('profile_id')['electricity_demand_normalised'].sum().reset_index()

	half_hourly_df['electricity_demand_normalised'] = half_hourly_df['electricity_demand_normalised'] / profile_grouped_df['electricity_demand_normalised'].sum()
	half_hourly_df['hh_period'] = ((half_hourly_df['hour']*2)+(half_hourly_df['minute']/30.)).astype(int)
	half_hourly_df['day_of_week'] = half_hourly_df['datetime'].dt.dayofweek

	half_hourly_df['electricity_demand_modelled_Wh'] = half_hourly_df['electricity_demand_normalised']*(annual_electricity_consumption_kWh*1000.)


	half_hourly_df['periods_since_monday_home_arrival'] = (half_hourly_df['day_of_week']*48)+(half_hourly_df['hh_period'])-home_arrival_hh_period

	half_hourly_df.loc[half_hourly_df['periods_since_monday_home_arrival']<0, 'periods_since_monday_home_arrival'] = half_hourly_df.loc[half_hourly_df['periods_since_monday_home_arrival']<0, 'periods_since_monday_home_arrival']+336

	return half_hourly_df

@st.cache_data
def get_hourly_PVGIS_file(latitude, longitude, azimuth, tilt):
	
	output = pvlib.iotools.get_pvgis_hourly(latitude, longitude, start=2019, end=2019, 
								   raddatabase='PVGIS-SARAH2', components=True, 
								   surface_tilt=tilt, 
								   surface_azimuth=azimuth, outputformat='json', usehorizon=True, 
								   userhorizon=None, pvcalculation=True, peakpower=1., 
								   pvtechchoice='crystSi', mountingplace='free', loss=14, 
								   trackingtype=0, optimal_surface_tilt=False, optimalangles=False, 
								   url='https://re.jrc.ec.europa.eu/api/v5_2/', map_variables=True, timeout=30)
	pvgis_output_df = output[0].reset_index().fillna(0)

	pvgis_output_df['datetime'] = pd.to_datetime(pvgis_output_df['time'], format='%Y%m%d:%H%M').dt.round('h').dt.tz_localize(None)
	pvgis_output_df.drop(labels=['time','poa_direct','poa_sky_diffuse','poa_ground_diffuse','solar_elevation','Int'], inplace=True, axis=1)
	pvgis_output_df.rename(columns={'P':'watts_per_kWp',
					   'temp_air':'temperature_2m_degC'
					   },inplace=True)

	pvgis_output_df['datetime'] = pd.to_datetime(pvgis_output_df['datetime'])
	pvgis_output_df = pvgis_output_df.set_index('datetime')
	pvgis_output_df = pvgis_output_df.resample('30min').interpolate().reset_index()
	pvgis_output_df['day_of_year'] = pvgis_output_df['datetime'].dt.dayofyear
	pvgis_output_df['hour'] = pvgis_output_df['datetime'].dt.hour
	pvgis_output_df['minute'] = pvgis_output_df['datetime'].dt.minute    

	return pvgis_output_df



def calculate_heat_demand(df, user_gas_demand_kWh):
	temperature_daily_df = df.groupby('day_of_year')['temperature_2m_degC'].mean().reset_index()

	# Estimating Daily Gas Demand based on outdoor temperature and Annual Gas demand
	# Using empirical estimator from S.D.Watson et al, https://www.sciencedirect.com/science/article/pii/S0301421518307249

	temperature_daily_df = temperature_daily_df.loc[temperature_daily_df['day_of_year']<=365]

	temperature_daily_df['daily_gas_demand_kWh_raw'] = 0.
	cond1 = (temperature_daily_df['temperature_2m_degC']<14.2)
# 	temperature_daily_df['daily_gas_demand_kWh_raw'].loc[cond1] = (-5.463*temperature_daily_df['temperature_2m_degC'].loc[cond1])+90.55
	temperature_daily_df.loc[cond1, 'daily_gas_demand_kWh_raw'] = (-5.463*temperature_daily_df['temperature_2m_degC'].loc[cond1])+90.55	

	cond2 = (temperature_daily_df['temperature_2m_degC']>=14.2)
# 	temperature_daily_df['daily_gas_demand_kWh_raw'].loc[cond2] = (-0.988*temperature_daily_df['temperature_2m_degC'].loc[cond2])+26.84
	temperature_daily_df.loc[cond2, 'daily_gas_demand_kWh_raw'] = (-0.988*temperature_daily_df['temperature_2m_degC'].loc[cond2])+26.84

	raw_gas_demand_kWh = temperature_daily_df['daily_gas_demand_kWh_raw'].sum()

	temperature_daily_df['daily_proportion_of_annual_gas_demand'] = (temperature_daily_df['daily_gas_demand_kWh_raw']/
																		 raw_gas_demand_kWh)

	print(temperature_daily_df['daily_proportion_of_annual_gas_demand'].sum())

	drop_cols = ['daily_gas_demand_kWh_raw']
	temperature_daily_df.drop(labels=drop_cols, axis=1, inplace=True)

	# temperature_daily_df['modelled_gas_demand_kWh'] = temperature_daily_df['daily_proportion_of_annual_gas_demand']*raw_gas_demand_kWh

	df = pd.merge(df, temperature_daily_df[['day_of_year','daily_proportion_of_annual_gas_demand']], on='day_of_year')
	df['modelled_heat_demand_Wh'] = user_gas_demand_kWh*1000.*df['daily_proportion_of_annual_gas_demand']/48.

	print (df['modelled_heat_demand_Wh'].sum())
	
	return df


def calculate_EV_charging_behaviour(rates_df_pivoted, annual_miles_driven, 
									ev_Wh_per_mile, ev_max_power_W, arrival_departure_delta_n_hh_periods):
									
	ev_demand_dict_list = []
	ev_Wh_annual_demand = annual_miles_driven*ev_Wh_per_mile
	ev_Wh_daily_demand = ev_Wh_annual_demand / 365.

	ev_max_Wh = ev_max_power_W*0.5

	n_ev_charge_periods = math.ceil(ev_Wh_daily_demand/ev_max_Wh)
	n_full_ev_charge_periods = math.floor(ev_Wh_daily_demand/ev_max_Wh)

	ev_partial_Wh = ev_Wh_daily_demand - (n_full_ev_charge_periods*ev_max_Wh)


	for tariff_id in rates_df_pivoted['tariff_id'].unique():

		ev_charging_Wh = np.array([])

		ev_charging_hh_periods_since_monday_home_arrival = np.array([])    
	#     Loop thru each "home arrival" row, starting with Monday
	#     Look for next "home departure" time
	#     Calculate the charging_opporunity_hh_periods for each time slot
	#     IF C is greater than B, add on however many slots needed to get us to full
	#     IF C is greater than A, add on however many slots needed to get us to full, then add warning flag
	#     Check conditions above.  If both pass, then 
# 		print (rates_df_pivoted.loc[(rates_df_pivoted['tariff_id']==tariff_id)&
# 								 (rates_df_pivoted['home_arrival_bool']==True)
# 								 ]['periods_since_monday_home_arrival'].values)
	
		for n in rates_df_pivoted.loc[(rates_df_pivoted['tariff_id']==tariff_id)&
								 (rates_df_pivoted['home_arrival_bool']==True)
								 ]['periods_since_monday_home_arrival'].values:
		
			charging_opporunity_hh_periods_preferred = rates_df_pivoted.loc[(rates_df_pivoted['tariff_id']==tariff_id)&
										(rates_df_pivoted['electricity_import_below_median_rate']==True)&
										(rates_df_pivoted['periods_since_monday_home_arrival']>=n)&
										(rates_df_pivoted['periods_since_monday_home_arrival']<n+arrival_departure_delta_n_hh_periods)
									   ]['periods_since_monday_home_arrival'].values

			charging_opporunity_hh_periods_not_preferred = rates_df_pivoted.loc[(rates_df_pivoted['tariff_id']==tariff_id)&
										(rates_df_pivoted['electricity_import_below_median_rate']==False)&
										(rates_df_pivoted['periods_since_monday_home_arrival']>=n)&
										(rates_df_pivoted['periods_since_monday_home_arrival']<n+arrival_departure_delta_n_hh_periods)
									   ]['periods_since_monday_home_arrival'].values
		

		
			if n_full_ev_charge_periods <= len(charging_opporunity_hh_periods_preferred):

	#             Full periods
				ev_charging_hh_periods_since_monday_home_arrival = np.append(ev_charging_hh_periods_since_monday_home_arrival, charging_opporunity_hh_periods_preferred[:n_full_ev_charge_periods])
				ev_charging_Wh = np.append(ev_charging_Wh, np.full_like(charging_opporunity_hh_periods_preferred[:n_full_ev_charge_periods], ev_max_Wh))
			
	#             Partial period
				ev_charging_hh_periods_since_monday_home_arrival = np.append(ev_charging_hh_periods_since_monday_home_arrival, charging_opporunity_hh_periods_preferred[n_full_ev_charge_periods:(n_full_ev_charge_periods+1)])
				ev_charging_Wh = np.append(ev_charging_Wh, np.array([ev_partial_Wh]))
		
			if n_full_ev_charge_periods > len(charging_opporunity_hh_periods_preferred):
	#             Full periods
				ev_charging_hh_periods_since_monday_home_arrival = np.append(ev_charging_hh_periods_since_monday_home_arrival, charging_opporunity_hh_periods_preferred[:n_full_ev_charge_periods])
				ev_charging_Wh = np.append(ev_charging_Wh, np.full_like(charging_opporunity_hh_periods_preferred[:n_full_ev_charge_periods], ev_max_Wh))

				remaining_hh_periods = n_full_ev_charge_periods - len(charging_opporunity_hh_periods_preferred)
			
			
				ev_charging_hh_periods_since_monday_home_arrival = np.append(ev_charging_hh_periods_since_monday_home_arrival, charging_opporunity_hh_periods_not_preferred[:remaining_hh_periods])
				ev_charging_Wh = np.append(ev_charging_Wh, np.full_like(charging_opporunity_hh_periods_not_preferred[:remaining_hh_periods], ev_max_Wh))            
						
	#             Partial period
				ev_charging_hh_periods_since_monday_home_arrival = np.append(ev_charging_hh_periods_since_monday_home_arrival, charging_opporunity_hh_periods_not_preferred[remaining_hh_periods:(remaining_hh_periods+1)])
				ev_charging_Wh = np.append(ev_charging_Wh, np.array([ev_partial_Wh]))
		

	
	
	
		ev_d = {'tariff_id':tariff_id,
				'periods_since_monday_home_arrival':list(ev_charging_hh_periods_since_monday_home_arrival),
				'ev_demand_Wh':list(ev_charging_Wh),
				'hh_period':list(rates_df_pivoted.loc[(rates_df_pivoted['tariff_id']==tariff_id)]['hh_period'].values),
				'day_of_week':list(rates_df_pivoted.loc[(rates_df_pivoted['tariff_id']==tariff_id)]['day_of_week'].values),
				'electricity_import_below_median_rate':list(rates_df_pivoted.loc[(rates_df_pivoted['tariff_id']==tariff_id)]['electricity_import_below_median_rate'].values),
				'electricity_export_above_median_rate':list(rates_df_pivoted.loc[(rates_df_pivoted['tariff_id']==tariff_id)]['electricity_export_above_median_rate'].values),				
				'electricity_import_unit_rate_per_kWh':list(rates_df_pivoted.loc[(rates_df_pivoted['tariff_id']==tariff_id)]['electricity_import_unit_rate_per_kWh'].values),
				'electricity_export_unit_rate_per_kWh':list(rates_df_pivoted.loc[(rates_df_pivoted['tariff_id']==tariff_id)]['electricity_export_unit_rate_per_kWh'].values),
				'gas_unit_rate_per_kWh':list(rates_df_pivoted.loc[(rates_df_pivoted['tariff_id']==tariff_id)]['gas_unit_rate_per_kWh'].values),				
			   }

		ev_demand_dict_list.append(ev_d)
		
	return ev_demand_dict_list



def expand_energy_tariffs(energy_tariffs_df, home_arrival_hh_period):
	# Converting & expanding Energy Tariff JSON into useful dataframes
	
	rates_df_list = []
	standing_charges_df_list = []
	for y in range(len(energy_tariffs_df.index)):
	
		for x in energy_tariffs_df.iloc[y]['tariff_rates']:
			if x['start_settlement_period'] > x['last_settlement_period']:
				settlement_periods = np.arange(x['start_settlement_period'],48,1)
				settlement_periods = np.append(settlement_periods, np.arange(0,x['last_settlement_period']+1,1))
			else:
				settlement_periods = np.arange(x['start_settlement_period'], x['last_settlement_period']+1, 1)
			days_of_week = np.array(x['days_of_week'])
			a = [settlement_periods,
				 days_of_week]

			combinations = list(itertools.product(*a))
			rate_df = pd.DataFrame(data=combinations, columns=['settlement_period',
														  'day_of_week'])
			rate_df['fuel'] = x['fuel']
			rate_df['direction'] = x['direction']
			rate_df['period_name'] = x['period_name']    
			rate_df['unit_rate'] = x['unit_rate']
			rate_df['tariff_id'] = energy_tariffs_df.iloc[y]['tariff_id']
			rate_df['tariff_type'] = energy_tariffs_df.iloc[y]['tariff_type']        
			rates_df_list.append(rate_df)

		standing_charge_df = pd.DataFrame(data=energy_tariffs_df.iloc[y]['tariff_standing_charge'])
		standing_charge_df['tariff_id'] = energy_tariffs_df.iloc[y]['tariff_id']
		standing_charges_df_list.append(standing_charge_df)
	

	rates_df = pd.concat(rates_df_list)
	standing_charges_df = pd.concat(standing_charges_df_list)

	import_rates_df_pivoted = pd.pivot_table(rates_df.loc[rates_df['direction']=='import'], 
											 values='unit_rate', index=['settlement_period', 'day_of_week',
																		'tariff_id','tariff_type'],
						columns=['fuel'], aggfunc=np.sum).reset_index()

	export_rates_df_pivoted = pd.pivot_table(rates_df.loc[rates_df['direction']=='export'], 
											 values='unit_rate', index=['settlement_period', 'day_of_week',
																		'tariff_id','tariff_type'],
						columns=['fuel'], aggfunc=np.sum).reset_index()
	

	export_rates_df_pivoted.rename(columns={'electricity':'electricity_export_unit_rate_per_kWh',
											'settlement_period':'hh_period'}, inplace=True)


	import_rates_df_pivoted.rename(columns={'electricity':'electricity_import_unit_rate_per_kWh',
											'gas':'gas_unit_rate_per_kWh',
											'settlement_period':'hh_period'}, inplace=True)

# 	calculate the average electricity import rate for each tariff, for each fuel

	avg_import_rates_df = import_rates_df_pivoted.groupby(['tariff_id','tariff_type'])[['electricity_import_unit_rate_per_kWh','gas_unit_rate_per_kWh']].median().reset_index()

	avg_import_rates_df.rename(columns={'electricity_import_unit_rate_per_kWh':'median_electricity_unit_rate_per_kWh',
								 'gas_unit_rate_per_kWh':'median_gas_unit_rate_per_kWh'}, inplace=True)
								 
	import_rates_df_pivoted = pd.merge(import_rates_df_pivoted, avg_import_rates_df,
								on=['tariff_id','tariff_type'])

	
# 	calculate the average electricity export rate for each tariff, for each fuel
	
	avg_export_rates_df = export_rates_df_pivoted.groupby(['tariff_id','tariff_type'])[['electricity_export_unit_rate_per_kWh']].median().reset_index()

	avg_export_rates_df.rename(columns={'electricity_export_unit_rate_per_kWh':'median_electricity_export_unit_rate_per_kWh'},
							   inplace=True)

	export_rates_df_pivoted = pd.merge(export_rates_df_pivoted, avg_export_rates_df,
								on=['tariff_id','tariff_type'])


# 	Calculate when export rates are above average (median)
	export_rates_df_pivoted['electricity_export_above_median_rate'] = False
	cond = (export_rates_df_pivoted['electricity_export_unit_rate_per_kWh']>export_rates_df_pivoted['median_electricity_export_unit_rate_per_kWh'])
	export_rates_df_pivoted.loc[cond,'electricity_export_above_median_rate'] = True
	export_rates_df_pivoted.drop(labels=['median_electricity_export_unit_rate_per_kWh'],
						  axis=1,inplace=True)

# 	Calculate when import rates are below average (median)
	import_rates_df_pivoted['electricity_import_below_median_rate'] = False
	cond = (import_rates_df_pivoted['electricity_import_unit_rate_per_kWh']<import_rates_df_pivoted['median_electricity_unit_rate_per_kWh'])
	import_rates_df_pivoted.loc[cond,'electricity_import_below_median_rate'] = True

	import_rates_df_pivoted['gas_at_or_below_mean_rate'] = False
	cond = (import_rates_df_pivoted['gas_unit_rate_per_kWh']<import_rates_df_pivoted['median_gas_unit_rate_per_kWh'])
	import_rates_df_pivoted.loc[cond, 'gas_at_or_below_mean_rate'] = True

	import_rates_df_pivoted.drop(labels=['median_electricity_unit_rate_per_kWh','median_gas_unit_rate_per_kWh'],
						  axis=1,inplace=True)

	# import_rates_df_pivoted.rename(columns={'settlement_period':'hh_period'},inplace=True)

	import_rates_df_pivoted['home_arrival_bool'] = False

	import_rates_df_pivoted.loc[import_rates_df_pivoted['hh_period']==home_arrival_hh_period,'home_arrival_bool'] = True


	import_rates_df_pivoted['periods_since_last_home_arrival'] =  import_rates_df_pivoted['hh_period']- home_arrival_hh_period

	import_rates_df_pivoted.loc[import_rates_df_pivoted['periods_since_last_home_arrival']<0, 'periods_since_last_home_arrival'] = 48+import_rates_df_pivoted['periods_since_last_home_arrival'].loc[import_rates_df_pivoted['periods_since_last_home_arrival']<0]

	import_rates_df_pivoted['periods_since_monday_home_arrival'] = (import_rates_df_pivoted['day_of_week']*48)+(import_rates_df_pivoted['hh_period'])-home_arrival_hh_period

	import_rates_df_pivoted.loc[import_rates_df_pivoted['periods_since_monday_home_arrival']<0, 'periods_since_monday_home_arrival'] = import_rates_df_pivoted['periods_since_monday_home_arrival'].loc[import_rates_df_pivoted['periods_since_monday_home_arrival']<0]+336

	
	rates_df_pivoted = pd.merge(import_rates_df_pivoted, export_rates_df_pivoted[['tariff_id','day_of_week','hh_period','electricity_export_above_median_rate','electricity_export_unit_rate_per_kWh']],
							   on=['tariff_id','day_of_week','hh_period'])

	rates_df_pivoted.sort_values(by=['tariff_id','periods_since_monday_home_arrival'],inplace=True)
	
	standing_charges_df_pivoted = pd.pivot_table(standing_charges_df, values='cost', index=['tariff_id'],
						columns=['fuel'], aggfunc=np.sum).reset_index()
	standing_charges_df_pivoted.index.name = None
	standing_charges_df_pivoted.rename(columns={'electricity':'electricity_standing_charge_daily',
									 'gas':'gas_standing_charge_daily'}, inplace=True)

	energy_tariffs_df = pd.merge(energy_tariffs_df, standing_charges_df_pivoted, on='tariff_id')
	
	return energy_tariffs_df, rates_df_pivoted


def create_scenarios(vehicles_df, battery_storage_systems_df, battery_units_df,
					 heating_systems_df, solar_pv_systems_df, 
					 # solar_power_df,
					 ev_chargers_df, energy_tariffs_df, ev_demand_dict_list, 
					 technology_choices_dict):

	heating_systems_df = heating_systems_df.loc[heating_systems_df['heating_system_name'].isin([technology_choices_dict['current_heating_system']]+[technology_choices_dict['future_heating_system']])]
	battery_storage_systems_df = battery_storage_systems_df.loc[battery_storage_systems_df['battery_storage_name'].isin([technology_choices_dict['current_battery_storage_system']]+[technology_choices_dict['battery_storage_option']])]			
	solar_power_df = pd.DataFrame(data={'solar_pv_power_kWp':[0,technology_choices_dict['future_solar_pv_power_Wp']]})
	solar_power_df['solar_pv_power_kWp'] = solar_power_df['solar_pv_power_kWp'] / 1000.
	solar_pv_systems_df = solar_pv_systems_df.loc[solar_pv_systems_df['solar_pv_name'].isin([technology_choices_dict['current_solar_pv_system']]+[technology_choices_dict['solar_pv_option']])]			
	energy_tariffs_df = energy_tariffs_df.loc[energy_tariffs_df['tariff_name'].isin([technology_choices_dict['current_energy_tariff']]+[technology_choices_dict['energy_tariff_option']])]
	vehicles_df = vehicles_df.loc[vehicles_df['vehicle_name'].isin([technology_choices_dict['current_vehicle']]+[technology_choices_dict['future_vehicle']])]

# 	Filter out No solar PV, but size >0
	scenario_df = pd.merge(vehicles_df, battery_storage_systems_df, how='cross')
	scenario_df = pd.merge(scenario_df, battery_units_df, how='cross')
	scenario_df = pd.merge(scenario_df, heating_systems_df, how='cross')
	scenario_df = pd.merge(scenario_df, solar_pv_systems_df, how='cross')
	scenario_df = pd.merge(scenario_df, solar_power_df, how='cross')
	scenario_df = pd.merge(scenario_df, ev_chargers_df, how='cross')
	scenario_df = pd.merge(scenario_df, energy_tariffs_df[['tariff_id','tariff_name','tariff_requires_smart_meter',
														   'electricity_standing_charge_daily',
														   'gas_standing_charge_daily']], how='cross')



	drop_cond = (((scenario_df['solar_pv_name'] == 'No Solar PV') & (scenario_df['solar_pv_power_kWp']>0.)) |
				 ((scenario_df['solar_pv_name'] != 'No Solar PV') & (scenario_df['solar_pv_power_kWp']==0.)) )

	scenario_df.drop(scenario_df[drop_cond].index, inplace=True)
	
# 	Set all combinations  with no battery storage to battery_num_units and de-duplicate

	no_battery_cond = (scenario_df['battery_storage_name'] == 'No Battery Storage')
	
	scenario_df.loc[no_battery_cond, 'battery_num_units'] = 0
		
	drop_cond = ((scenario_df['battery_storage_name'] == 'No Battery Storage') & (scenario_df['battery_num_units'] > 1))

	scenario_df.drop(scenario_df[drop_cond].index, inplace=True)
	

	scenario_df['battery_storage_cost'] = (scenario_df['battery_storage_unit_cost'] * 
										   scenario_df['battery_num_units'])

	scenario_df['scenario_id'] = [n for n in range(len(scenario_df.index))]


	scenarios_dict = scenario_df.to_dict('records')
	list_comp = [d['tariff_id'] for d in ev_demand_dict_list]
	
	for n in range(len(scenarios_dict)):
		
		t_id = scenarios_dict[n]['tariff_id']
		idx = list_comp.index(t_id)

		scenarios_dict[n]['ev_hh_periods_since_monday_home_arrival'] = ev_demand_dict_list[idx]['periods_since_monday_home_arrival']
		scenarios_dict[n]['ev_demand_Wh'] = ev_demand_dict_list[idx]['ev_demand_Wh']    
		scenarios_dict[n]['rates_hh_period'] = ev_demand_dict_list[idx]['hh_period']    
		scenarios_dict[n]['rates_day_of_week'] = ev_demand_dict_list[idx]['day_of_week']    
		scenarios_dict[n]['rates_electricity_import_below_median_rate'] = ev_demand_dict_list[idx]['electricity_import_below_median_rate']    
		scenarios_dict[n]['rates_electricity_export_above_median_rate'] = ev_demand_dict_list[idx]['electricity_export_above_median_rate']    		
		scenarios_dict[n]['rate_electricity_import_unit_rate_per_kWh'] = ev_demand_dict_list[idx]['electricity_import_unit_rate_per_kWh']    
		scenarios_dict[n]['rate_electricity_export_unit_rate_per_kWh'] = ev_demand_dict_list[idx]['electricity_export_unit_rate_per_kWh']				
		scenarios_dict[n]['rates_gas_unit_rate_per_kWh'] = ev_demand_dict_list[idx]['gas_unit_rate_per_kWh']
		scenarios_dict[n]['export_limit_kW'] = technology_choices_dict['export_limit_kW']

		
	return scenario_df, scenarios_dict


def calculate_energy_balance(input_df, params):

    grid_elec_import_Wh_list = []
    grid_elec_export_Wh_list = []
    modelled_gas_demand_list = []
    ev_charging_demand_Wh_list = []
    electricity_demand_heatpump_Wh_list = []
    solar_pv_generation_Wh_list = []

    battery_generation_Wh_list, battery_charging_demand_Wh_list, battery_energy_stored_energy_Wh_beginning_of_period_list, battery_energy_stored_energy_Wh_end_of_period_list, pv_satisfy_battery_demand_Wh_list = ([] for i in range(5))
    ev_energy_stored_energy_Wh_beginning_of_period = 0.
    ev_energy_stored_energy_Wh_end_of_period = 0.
    battery_energy_stored_energy_Wh_beginning_of_period = 0.
    battery_energy_stored_energy_Wh_end_of_period = 0.
    
    ev_df = pd.DataFrame(data={'periods_since_monday_home_arrival':params['ev_hh_periods_since_monday_home_arrival'],
                               'ev_demand_Wh':params['ev_demand_Wh']})
    
    rates_df = pd.DataFrame(data={'hh_period':params['rates_hh_period'],
                                  'day_of_week':params['rates_day_of_week'],
                                  'electricity_import_below_median_rate':params['rates_electricity_import_below_median_rate'],
                                  'electricity_export_above_median_rate':params['rates_electricity_export_above_median_rate'],
                                  'electricity_import_unit_rate_per_kWh':params['rate_electricity_import_unit_rate_per_kWh'],
                                  'electricity_export_unit_rate_per_kWh':params['rate_electricity_export_unit_rate_per_kWh'],
                                  'gas_unit_rate_per_kWh':params['rates_gas_unit_rate_per_kWh']
                                  })
    
    input_df = pd.merge(input_df, ev_df, on='periods_since_monday_home_arrival',how='left').sort_values(by='datetime', ascending=True)
    input_df.fillna({'ev_demand_Wh':0}, inplace=True)
    
    input_df = pd.merge(input_df, rates_df,on=['day_of_week','hh_period']).sort_values(by='datetime', ascending=True)

    if params['vehicle_fuel_type'] == 'gasoline':
        input_df['ev_demand_Wh'] = 0.
    
    # PV Generation is now dependent on the scenario!
    input_df['pv_generation_Wh'] = input_df['watts_per_kWp']*params['solar_pv_power_kWp']*0.5
    if params['heating_system_fuel_type'] == 'electricity':
        input_df['heatpump_demand_Wh'] = input_df['modelled_heat_demand_Wh']/params['heating_system_efficiency']
        input_df['modelled_gas_demand_Wh'] = 0.
    else:
        input_df['heatpump_demand_Wh'] = 0.
        input_df['modelled_gas_demand_Wh'] = input_df['modelled_heat_demand_Wh']/params['heating_system_efficiency']
    
    input_df['non_battery_demand_Wh'] = (input_df['electricity_demand_modelled_Wh']+
                                   input_df['ev_demand_Wh']+
                                   input_df['heatpump_demand_Wh'])
    modelled_gas_demand_list = list(input_df['modelled_gas_demand_Wh'].values)
    
    ev_charging_demand_Wh_list = list(input_df['ev_demand_Wh'].values)
    electricity_demand_heatpump_Wh_list = list(input_df['heatpump_demand_Wh'].values)
    solar_pv_generation_Wh_list = list(input_df['pv_generation_Wh'].values)
             
    electricity_unit_rate_per_kWh_list = list(input_df['electricity_import_unit_rate_per_kWh'].values)
    electricity_export_unit_rate_per_kWh_list = list(input_df['electricity_export_unit_rate_per_kWh'].values) 
    gas_unit_rate_per_kWh_list = list(input_df['gas_unit_rate_per_kWh'].values)
    
    if max(electricity_export_unit_rate_per_kWh_list) == min(electricity_export_unit_rate_per_kWh_list):
        variable_electricity_export_rate = False
    else:
        variable_electricity_export_rate = True

    battery_number_units = params['battery_num_units']
    battery_energy_max_capacity_Wh = params['battery_storage_capacity_Wh']
    battery_storage_max_charge_rate_watts = params['battery_storage_max_charge_rate_watts']
    battery_storage_max_discharge_rate_watts = params['battery_storage_max_discharge_rate_watts']    
    battery_max_input_Wh = battery_number_units*battery_storage_max_charge_rate_watts*0.5
    battery_max_output_Wh = min(params['export_limit_kW']*1000.*0.5, battery_number_units*battery_storage_max_discharge_rate_watts*0.5)
    
    for n in range(len(input_df.index)):
        battery_energy_stored_energy_Wh_beginning_of_period = battery_energy_stored_energy_Wh_end_of_period        
        battery_energy_until_full_Wh_start_of_period = (battery_number_units * battery_energy_max_capacity_Wh) - battery_energy_stored_energy_Wh_beginning_of_period
        ev_energy_stored_energy_Wh_beginning_of_period = ev_energy_stored_energy_Wh_end_of_period        
        
        non_battery_demand_Wh = input_df['non_battery_demand_Wh'].values[n]
        pv_generation_Wh = input_df['pv_generation_Wh'].values[n]
        
        non_battery_demand_minus_pv_gen_Wh = non_battery_demand_Wh - pv_generation_Wh
        non_battery_demand_after_pv_gen_Wh = max(non_battery_demand_minus_pv_gen_Wh, 0)
        pv_satisfy_non_battery_demand_Wh = min(non_battery_demand_Wh, pv_generation_Wh)        
        pv_excess_generation_Wh = pv_generation_Wh - pv_satisfy_non_battery_demand_Wh
        
    
        battery_charging_demand_Wh = 0.
        battery_generation_Wh = 0.
        
        if pv_excess_generation_Wh > 0.:
            if input_df['electricity_export_above_median_rate'].values[n]:
                battery_charging_demand_Wh = 0.
            else:	
                battery_charging_demand_Wh = min(battery_energy_until_full_Wh_start_of_period,
												 battery_max_input_Wh, 
												 pv_excess_generation_Wh)
        else:

            if input_df['electricity_import_below_median_rate'].values[n]:
                battery_charging_demand_Wh = min(battery_energy_until_full_Wh_start_of_period,
												 battery_max_input_Wh)
            elif input_df['electricity_export_above_median_rate'].values[n]:
                battery_generation_Wh = min(battery_max_output_Wh,
							battery_energy_stored_energy_Wh_beginning_of_period)
            else:
                if not variable_electricity_export_rate:
                    battery_generation_Wh = min(battery_max_output_Wh,
								battery_energy_stored_energy_Wh_beginning_of_period)
									
                else:
                    battery_generation_Wh = min(non_battery_demand_minus_pv_gen_Wh,
								battery_energy_stored_energy_Wh_beginning_of_period)


# 		If there is excess PV generation that isn't taken up by loads, we should:
# 			B) Export the excess generation, IF the EXPORT rate is peak (this is an uncommon edge case)
# 			A) Use this excess to charge the battery, IF the EXPORT rate is not peak
# 		Else...(if there is no excess PV generation)....
# 		If the IMPORT rate is off-peak, we should charge the battery, and not discharge
# 		If the EXPORT rate is peak, we should discharge the battery, and not charge

        pv_satisfy_battery_demand_Wh = min(battery_charging_demand_Wh, pv_excess_generation_Wh)
        pv_export_Wh = pv_generation_Wh - pv_satisfy_non_battery_demand_Wh - pv_satisfy_battery_demand_Wh
		
# 		For grid_net_flow_Wh, positive values represent battery discharging, negative values represent battery charging
        battery_net_flow_Wh = battery_generation_Wh - battery_charging_demand_Wh
		
# 		For grid_net_flow_Wh, positive values represent import into property from the grid, negative values represent export from property into the grid
        grid_net_flow_Wh = non_battery_demand_Wh - pv_generation_Wh - battery_net_flow_Wh
		
        grid_elec_import_Wh = max(0, grid_net_flow_Wh)
		
        grid_elec_export_Wh = -min(0, grid_net_flow_Wh)

        battery_energy_stored_energy_Wh_end_of_period = (battery_energy_stored_energy_Wh_beginning_of_period +
                                                          battery_charging_demand_Wh - 
                                                          battery_generation_Wh
                                                         )
        
        grid_elec_import_Wh_list.append(grid_elec_import_Wh)
        grid_elec_export_Wh_list.append(grid_elec_export_Wh)
        battery_generation_Wh_list.append(battery_generation_Wh)
        battery_charging_demand_Wh_list.append(battery_charging_demand_Wh)

        battery_energy_stored_energy_Wh_beginning_of_period_list.append(battery_energy_stored_energy_Wh_beginning_of_period)
        battery_energy_stored_energy_Wh_end_of_period_list.append(battery_energy_stored_energy_Wh_end_of_period)
        pv_satisfy_battery_demand_Wh_list.append(pv_satisfy_battery_demand_Wh)
	# Design some checksums - this should add up to zero (or v close)
    checksum = (input_df['ev_demand_Wh'].sum()+
                input_df['electricity_demand_modelled_Wh'].sum()+
                input_df['heatpump_demand_Wh'].sum()+
                sum(battery_charging_demand_Wh_list)-
                sum(battery_generation_Wh_list)-
                input_df['pv_generation_Wh'].sum()-
                sum(grid_elec_import_Wh_list)+
                sum(grid_elec_export_Wh_list)
                )

    return grid_elec_import_Wh_list, grid_elec_export_Wh_list, battery_generation_Wh_list, battery_charging_demand_Wh_list, battery_energy_stored_energy_Wh_beginning_of_period_list, battery_energy_stored_energy_Wh_end_of_period_list, pv_satisfy_battery_demand_Wh_list, electricity_unit_rate_per_kWh_list, electricity_export_unit_rate_per_kWh_list, gas_unit_rate_per_kWh_list, modelled_gas_demand_list, ev_charging_demand_Wh_list, electricity_demand_heatpump_Wh_list, solar_pv_generation_Wh_list


def process_scenarios(input_df, scenario_df, scenarios_dict):

    t_start = time.time()
    summary_results_df_list = []

    for target_scenario_id in scenario_df['scenario_id'].unique():

        print (target_scenario_id)

        grid_elec_import_Wh_list, grid_elec_export_Wh_list, battery_generation_Wh_list, battery_charging_demand_Wh_list, battery_energy_stored_energy_Wh_beginning_of_period_list, battery_energy_stored_energy_Wh_end_of_period_list, pv_satisfy_battery_demand_Wh_list, electricity_unit_rate_per_kWh_list, electricity_export_unit_rate_per_kWh_list, gas_unit_rate_per_kWh_list, modelled_gas_demand_list, ev_charging_demand_Wh_list, electricity_demand_heatpump_Wh_list, solar_pv_generation_Wh_list = calculate_energy_balance(input_df, scenarios_dict[target_scenario_id])
		
        results_df = pd.DataFrame(data={'datetime':df['datetime'].values})
        results_df['baseload_demand_Wh'] = input_df['electricity_demand_modelled_Wh'].values
        results_df['hh_period'] = input_df['hh_period'].values
        results_df['scenario_id'] = target_scenario_id
        results_df['grid_elec_import_Wh'] = grid_elec_import_Wh_list
        results_df['grid_elec_export_Wh'] = grid_elec_export_Wh_list
        results_df['battery_generation_Wh'] = battery_generation_Wh_list
        results_df['battery_charging_demand_Wh'] = battery_charging_demand_Wh_list
        results_df['battery_energy_stored_energy_Wh_beginning_of_period'] = battery_energy_stored_energy_Wh_beginning_of_period_list
        results_df['battery_energy_stored_energy_Wh_end_of_period'] = battery_energy_stored_energy_Wh_end_of_period_list
        results_df['pv_satisfy_battery_demand_Wh'] = pv_satisfy_battery_demand_Wh_list
        results_df['modelled_gas_demand_Wh'] = modelled_gas_demand_list
        results_df['electricity_import_unit_rate_per_kWh'] = electricity_unit_rate_per_kWh_list
        results_df['gas_unit_rate_per_kWh'] = gas_unit_rate_per_kWh_list
        results_df['electricity_export_unit_rate_per_kWh'] = electricity_export_unit_rate_per_kWh_list
        results_df['electricity_import_cost'] = results_df['grid_elec_import_Wh']*results_df['electricity_import_unit_rate_per_kWh']/1000.
        results_df['gas_import_cost'] = results_df['modelled_gas_demand_Wh']*results_df['gas_unit_rate_per_kWh']/1000.
        results_df['electricity_export_revenue'] = results_df['grid_elec_export_Wh']*results_df['electricity_export_unit_rate_per_kWh']/1000.    
        results_df['ev_charging_demand_Wh'] = ev_charging_demand_Wh_list
        results_df['electricity_demand_heatpump_Wh'] = electricity_demand_heatpump_Wh_list
        results_df['solar_pv_generation_Wh'] = solar_pv_generation_Wh_list
        
        agg_cols = ['grid_elec_import_Wh','grid_elec_export_Wh','electricity_import_cost','gas_import_cost','electricity_export_revenue','ev_charging_demand_Wh','electricity_demand_heatpump_Wh','solar_pv_generation_Wh','baseload_demand_Wh']
		
        summary_results_df = results_df.groupby('scenario_id')[agg_cols].sum().reset_index()
        summary_results_df_list.append(summary_results_df)

    print ('Time to run loop:',time.time() - t_start,'seconds')
    summary_results_df = pd.concat(summary_results_df_list)    
    return summary_results_df

def evaluate_scenario(input_df, scenarios_dict, target_scenario_id):

    grid_elec_import_Wh_list, grid_elec_export_Wh_list, battery_generation_Wh_list, battery_charging_demand_Wh_list, battery_energy_stored_energy_Wh_beginning_of_period_list, battery_energy_stored_energy_Wh_end_of_period_list, pv_satisfy_battery_demand_Wh_list, electricity_unit_rate_per_kWh_list, electricity_export_unit_rate_per_kWh_list, gas_unit_rate_per_kWh_list, modelled_gas_demand_list, ev_charging_demand_Wh_list, electricity_demand_heatpump_Wh_list, solar_pv_generation_Wh_list = calculate_energy_balance(input_df, scenarios_dict)
		
    results_df = pd.DataFrame(data={'datetime':df['datetime'].values})
    results_df['baseload_demand_Wh'] = input_df['electricity_demand_modelled_Wh'].values
    results_df['hh_period'] = input_df['hh_period'].values
    results_df['scenario_id'] = target_scenario_id
    results_df['grid_elec_import_Wh'] = grid_elec_import_Wh_list
    results_df['grid_elec_export_Wh'] = grid_elec_export_Wh_list
    results_df['battery_generation_Wh'] = battery_generation_Wh_list
    results_df['battery_charging_demand_Wh'] = battery_charging_demand_Wh_list
    results_df['battery_energy_stored_energy_Wh_beginning_of_period'] = battery_energy_stored_energy_Wh_beginning_of_period_list
    results_df['battery_energy_stored_energy_Wh_end_of_period'] = battery_energy_stored_energy_Wh_end_of_period_list
    results_df['pv_satisfy_battery_demand_Wh'] = pv_satisfy_battery_demand_Wh_list
    results_df['modelled_gas_demand_Wh'] = modelled_gas_demand_list
    results_df['electricity_import_unit_rate_per_kWh'] = electricity_unit_rate_per_kWh_list
    results_df['gas_unit_rate_per_kWh'] = gas_unit_rate_per_kWh_list
    results_df['electricity_export_unit_rate_per_kWh'] = electricity_export_unit_rate_per_kWh_list
    results_df['electricity_import_cost'] = results_df['grid_elec_import_Wh']*results_df['electricity_import_unit_rate_per_kWh']/1000.
    results_df['gas_import_cost'] = results_df['modelled_gas_demand_Wh']*results_df['gas_unit_rate_per_kWh']/1000.
    results_df['electricity_export_revenue'] = results_df['grid_elec_export_Wh']*results_df['electricity_export_unit_rate_per_kWh']/1000.    
    results_df['ev_charging_demand_Wh'] = ev_charging_demand_Wh_list
    results_df['electricity_demand_heatpump_Wh'] = electricity_demand_heatpump_Wh_list
    results_df['solar_pv_generation_Wh'] = solar_pv_generation_Wh_list

    results_df['time_of_day'] = results_df['datetime'].dt.strftime('%H:%M')

    return results_df

@st.dialog("Detailed Analysis", width="large")
def generate_detailed_analysis(current_scenario_id, future_scenario_id):

	selected_future_scenario_cond = (summary_results_df['scenario_id']==future_scenario_id)	
	current_scenario_cond = (summary_results_df['scenario_id']==current_scenario_id)	

	future_solar_pv_system = summary_results_df['solar_pv_name'].loc[selected_future_scenario_cond].values[0]

	future_elec_standing_charge = summary_results_df.loc[selected_future_scenario_cond]['electricity_standing_charge_annual'].values[0]

	future_elec_cost = (summary_results_df.loc[selected_future_scenario_cond]['electricity_import_cost'].values[0] 
						- summary_results_df.loc[selected_future_scenario_cond]['electricity_export_revenue'].values[0]
						+ summary_results_df.loc[selected_future_scenario_cond]['electricity_standing_charge_annual'].values[0])
					
	future_gas_cost = (summary_results_df.loc[selected_future_scenario_cond]['gas_import_cost'].values[0]
					   + summary_results_df.loc[selected_future_scenario_cond]['gas_standing_charge_annual'].values[0])
				   
	future_ice_fuel_cost = summary_results_df.loc[selected_future_scenario_cond]['vehicle_fuel_cost'].values[0]

	future_total_energy_cost = (future_elec_cost+future_gas_cost+future_ice_fuel_cost)



	future_elec_effective_cost_per_kWh = summary_results_df.loc[selected_future_scenario_cond]['electricity_import_cost'].values[0]/(summary_results_df.loc[selected_future_scenario_cond]['grid_elec_import_Wh'].values[0]*0.001)
	
	future_baseload_demand_Wh = current_baseload_demand_Wh
	future_ev_demand_Wh = summary_results_df.loc[selected_future_scenario_cond]['ev_charging_demand_Wh'].values[0]
	future_heatpump_demand_Wh = summary_results_df.loc[selected_future_scenario_cond]['electricity_demand_heatpump_Wh'].values[0]
	future_solar_pv_generation_Wh = summary_results_df.loc[selected_future_scenario_cond]['solar_pv_generation_Wh'].values[0]
	future_grid_elec_export_Wh = summary_results_df.loc[selected_future_scenario_cond]['grid_elec_export_Wh'].values[0]
	future_solar_pv_self_consumed_Wh = future_solar_pv_generation_Wh - future_grid_elec_export_Wh
	future_total_demand_Wh = future_baseload_demand_Wh + future_ev_demand_Wh + future_heatpump_demand_Wh
	future_grid_import_Wh = summary_results_df.loc[selected_future_scenario_cond]['grid_elec_import_Wh'].values[0]
	
	future_grid_elec_import_cost = summary_results_df.loc[selected_future_scenario_cond]['electricity_import_cost'].values[0]
	future_baseload_elec_cost = future_grid_elec_import_cost * (future_baseload_demand_Wh/(future_baseload_demand_Wh+future_ev_demand_Wh+future_heatpump_demand_Wh))

	future_ev_elec_cost = future_grid_elec_import_cost * (future_ev_demand_Wh/(future_baseload_demand_Wh+future_ev_demand_Wh+future_heatpump_demand_Wh))
	future_heatpump_elec_cost = future_grid_elec_import_cost * (future_heatpump_demand_Wh/(future_baseload_demand_Wh+future_ev_demand_Wh+future_heatpump_demand_Wh))
	future_elec_export_income = summary_results_df.loc[selected_future_scenario_cond]['electricity_export_revenue'].values[0]

	future_gas_standing_charge = summary_results_df.loc[selected_future_scenario_cond]['gas_standing_charge_annual'].values[0]
	future_gas_usage_cost = summary_results_df.loc[selected_future_scenario_cond]['gas_import_cost'].values[0]

	future_solar_pv_self_consumed_fraction = future_solar_pv_self_consumed_Wh / future_solar_pv_generation_Wh
	future_solar_pv_exported_fraction = 1.-future_solar_pv_self_consumed_fraction

	current_elec_export_income = summary_results_df.loc[current_scenario_cond]['electricity_export_revenue'].values[0]
	
	# Calculating product upgrades
	product_type = []
	product_from = []
	product_to = []
	product_investment_cost = []

	if summary_results_df.loc[current_cond]['heating_system_name'].values[0] != summary_results_df.loc[selected_future_scenario_cond]['heating_system_name'].values[0]:
		product_type.append('Heating System')
		product_from.append(summary_results_df.loc[current_cond]['heating_system_name'].values[0])
		product_to.append(summary_results_df.loc[selected_future_scenario_cond]['heating_system_name'].values[0])
		product_investment_cost.append(int(summary_results_df.loc[selected_future_scenario_cond]['heating_system_cost'].values[0]))

	if summary_results_df.loc[current_cond]['solar_pv_name'].values[0] != summary_results_df.loc[selected_future_scenario_cond]['solar_pv_name'].values[0]:
		product_type.append('Solar PV')
		product_from.append(summary_results_df.loc[current_cond]['solar_pv_name'].values[0])
		if summary_results_df.loc[current_cond]['solar_pv_power_kWp'].values[0] > 0.:
			product_from[-1] = product_from[-1] + ' - ' + str(summary_results_df.loc[current_cond]['solar_pv_power_kWp'].values[0])+'kWp'
		product_to.append(summary_results_df.loc[selected_future_scenario_cond]['solar_pv_name'].values[0])
		if summary_results_df.loc[selected_future_scenario_cond]['solar_pv_power_kWp'].values[0] > 0.:
			product_to[-1] = product_to[-1] + ' - ' + str(summary_results_df.loc[selected_future_scenario_cond]['solar_pv_power_kWp'].values[0])+'kWp'
		product_investment_cost.append(int(summary_results_df.loc[selected_future_scenario_cond]['solar_pv_cost'].values[0]))

	if summary_results_df.loc[current_cond]['battery_storage_name'].values[0] != summary_results_df.loc[selected_future_scenario_cond]['battery_storage_name'].values[0]:
		product_type.append('Battery Storage')
		product_from.append(summary_results_df.loc[current_cond]['battery_storage_name'].values[0])
		product_to.append(summary_results_df.loc[selected_future_scenario_cond]['battery_storage_name'].values[0])
		product_investment_cost.append(int(summary_results_df.loc[selected_future_scenario_cond]['battery_storage_cost'].values[0]))
		if summary_results_df.loc[selected_future_scenario_cond]['battery_num_units'].values[0] > 0.:
			product_to[-1] = product_to[-1] + '; ' + str(summary_results_df.loc[selected_future_scenario_cond]['battery_num_units'].values[0])+' unit(s)'


				
	if summary_results_df.loc[current_cond]['vehicle_name'].values[0] != summary_results_df.loc[selected_future_scenario_cond]['vehicle_name'].values[0]:
		product_type.append('Vehicle')
		product_from.append(summary_results_df.loc[current_cond]['vehicle_name'].values[0])
		product_to.append(summary_results_df.loc[selected_future_scenario_cond]['vehicle_name'].values[0])
		product_investment_cost.append(0)		
		
	if summary_results_df.loc[current_cond]['tariff_name'].values[0] != summary_results_df.loc[selected_future_scenario_cond]['tariff_name'].values[0]:

		product_type.append('Energy Tariff')
		product_from.append(summary_results_df.loc[current_cond]['tariff_name'].values[0])
		product_to.append(summary_results_df.loc[selected_future_scenario_cond]['tariff_name'].values[0])
		product_investment_cost.append(0)

	current_bill = current_ice_fuel_cost+current_gas_cost+current_elec_cost
	future_bill = future_ice_fuel_cost+future_gas_cost+future_elec_cost
	cost_savings = (current_ice_fuel_cost+current_gas_cost+current_elec_cost)-(future_ice_fuel_cost+future_gas_cost+future_elec_cost)


	elec_costs = [round(current_elec_cost,0), round(future_elec_cost,0)]
	elec_costs = [x if x!=0 else '-' for x in elec_costs]

	gas_costs = [round(current_gas_cost,0), round(future_gas_cost,0)]
	gas_costs = [x if x!=0 else '-' for x in gas_costs]


	ice_fuel_costs = [round(current_ice_fuel_cost,0), round(future_ice_fuel_cost,0)]
	ice_fuel_costs = [x if x!=0 else '-' for x in ice_fuel_costs]


	options = {
	  "legend": {},
	  "grid": {
		"left": '3%',
		"right": '4%',
		"bottom": '3%',
		"containLabel": True
	  },
		"yAxis": {
			"type": "value",
			"show": False,        
			},
		"xAxis": {
			"type": "category",
			"data": ["Current: £"+str(int(current_total_energy_cost)), "Future: £"+str(int(future_total_energy_cost))]
			},
		"series": [
			{
			  "name": 'Electricity',
			  "type": 'bar',
			  "stack": 'total',
			  "label": {"show": True},
			  "emphasis": {
				"focus": 'series'
				},
			  "data": elec_costs
			  },
			{
			  "name": 'Gas',
			  "type": 'bar',
			  "stack": 'total',
			  "label": {
				"show": True
			  },
			  "emphasis": {
				"focus": 'series'
			  },
			  "data": gas_costs
			},
			{
			  "name": 'Petrol/Diesel',
			  "type": 'bar',
			  "stack": 'total',
			  "label": {
				"show": True
			  },
			  "emphasis": {
				"focus": 'series'
			  },
			  "data": ice_fuel_costs
			}		
		]
	}
	
	

	changes_df = pd.DataFrame({
							   'Type': product_type, 
							   'From': product_from,
							   'To': product_to,
							   'Cost': product_investment_cost
							   })
		   
	
	col1, col2, col3 = st.columns([15,1,5])

	with col1:
		st.subheader('Product Upgrades')
		st.markdown("_Adjust investment cost manually by editing the numbers in the cell - payback and total investment figures will update_", unsafe_allow_html=False)

		show_cols = ['From','To','Cost']
		edited_df = st.data_editor(changes_df[show_cols], use_container_width=True, hide_index=True)

		st.markdown("*_Assumes same cost to lease EV as a Petrol/Diesel car_*", unsafe_allow_html=False)
		st.markdown("*_No cost associated with switching to new tariff, assumes smart meter can be installed free-of-charge_*", unsafe_allow_html=False)

	total_investment_cost = edited_df['Cost'].astype(int).sum()

	if cost_savings > 0:
		payback_years = int(np.ceil(total_investment_cost/cost_savings))
	else:
		payback_years = 0

	with col3:

		st.metric(label="Total Investment", value='£'+str(total_investment_cost))
		
		
		if cost_savings <= 0.:
			# st.metric(label=":red[Annual Loss]", value='£'+str(int(summary_results_df.loc[future_potential_cond].loc[selected_scenario_cond]['Annual Savings'].values[0])))				
			st.metric(label="Payback", value='N/A')
			
		else:
			# st.metric(label=":green[Annual Savings]", value='£'+str(int(summary_results_df.loc[future_potential_cond].loc[selected_scenario_cond]['Annual Savings'].values[0])))				
			st.metric(label="Payback", value=str(payback_years)+' years')	
		
		net_return_over_25_years = (cost_savings*25.)-total_investment_cost
		
		if net_return_over_25_years > 0:
			st.metric(label=":green[Net Return over 25 Years]", value='£'+str(int(round(net_return_over_25_years))))
		else:
			st.metric(label=":red[Net Return over 25 Years]", value='£'+str(int(round(net_return_over_25_years))))


	st.markdown("""---""")	

	st.subheader('Where are the savings coming from?')
	
	col3, col4 = st.columns([3,2])						   

	with col3:

		st_echarts(options=options)

	with col4:

		change_gas_cost = future_gas_cost-current_gas_cost
		change_elec_cost = future_elec_cost-current_elec_cost
		change_ice_fuel_cost = future_ice_fuel_cost-current_ice_fuel_cost	

		if change_elec_cost == 0:	
			st.write('Electricity - No Change')

		if change_elec_cost < 0:
			st.write('Electricity - Saving £'+"{:.2f}".format(abs(change_elec_cost))+'/year')
			st.write("- Energy Usage: £",summary_results_df.loc[current_cond]['electricity_import_cost'].values[0]," → ",future_grid_elec_import_cost)
			st.write("- Standing Charge: £",current_elec_standing_charge," → ",future_elec_standing_charge)
			st.write("- Export Income: £",summary_results_df.loc[current_cond]['electricity_export_revenue'].values[0]," → ",future_elec_export_income)						

			
		if change_elec_cost > 0:
			st.write('Electricity - Adding £'+"{:.2f}".format(abs(change_elec_cost))+'/year')
			st.write("- Energy Usage: £",summary_results_df.loc[current_cond]['electricity_import_cost'].values[0]," → ",future_grid_elec_import_cost)
			st.write("- Standing Charge: £",current_elec_standing_charge," → ",future_elec_standing_charge)
			st.write("- Export Income: £",summary_results_df.loc[current_cond]['electricity_export_revenue'].values[0]," → ",future_elec_export_income)						

		st.write("")
	
		if change_gas_cost < 0:
			st.write('Natural Gas - Saving £'+"{:.2f}".format(abs(change_gas_cost))+'/year')
			st.write("- Energy Usage: £",summary_results_df.loc[current_cond]['gas_import_cost'].values[0]," → ",future_gas_usage_cost)
			st.write("- Standing Charge: £",summary_results_df.loc[current_cond]['gas_standing_charge_annual'].values[0]," → ",future_gas_standing_charge)

		if change_gas_cost > 0:
			st.write('Natural Gas - Adding £'+"{:.2f}".format(abs(change_gas_cost))+'/year')
			st.write("- Energy Usage: £",summary_results_df.loc[current_cond]['gas_import_cost'].values[0]," → ",future_gas_usage_cost)
			st.write("- Standing Charge: £",summary_results_df.loc[current_cond]['gas_standing_charge_annual'].values[0]," → ",future_gas_standing_charge)

		if change_gas_cost == 0:	
			st.write('Natural Gas - No Change')

		st.write("")

		if change_ice_fuel_cost < 0:
			st.write('Petrol/Diesel - Saving £'+"{:.2f}".format(abs(change_ice_fuel_cost))+'/year')		

		if change_ice_fuel_cost > 0:
			st.write('Petrol/Diesel - Adding £'+"{:.2f}".format(abs(change_ice_fuel_cost))+'/year')
		
		if change_ice_fuel_cost == 0:	
			st.write('Petrol/Diesel - No Change')
				
	st.write("---")

	col10, col11 = st.columns(2)

	elec_st_charge = [round(current_elec_standing_charge,0), round(future_elec_standing_charge,0)]
	elec_st_charge = [x if x!=0 else '-' for x in elec_st_charge]
	elec_baseload = [round(current_baseload_elec_cost,0), round(future_baseload_elec_cost,0)]
	elec_baseload = [x if x!=0 else '-' for x in elec_baseload]
	elec_heatpump = [round(current_heatpump_elec_cost,0), round(future_heatpump_elec_cost,0)]
	elec_heatpump = [x if x!=0 else '-' for x in elec_heatpump]
	elec_ev = [round(current_ev_elec_cost,0), round(future_ev_elec_cost,0)]
	elec_ev = [x if x!=0 else '-' for x in elec_ev]
	gas_st_charge = [round(current_gas_standing_charge,0), round(future_gas_standing_charge,0)]
	gas_st_charge = [x if x!=0 else '-' for x in gas_st_charge]
	gas_usage = [round(current_gas_usage_cost,0), round(future_gas_usage_cost,0)]
	gas_usage = [x if x!=0 else '-' for x in gas_usage]
	ice_fuel = [round(current_ice_fuel_cost,0), round(future_ice_fuel_cost,0)]
	ice_fuel = [x if x!=0 else '-' for x in ice_fuel]
	elec_export = [-round(current_solar_pv_export_income,0), -round(future_elec_export_income,0)]
	elec_export = [x if x!=0 else '-' for x in elec_export]

	options_en_bill = {
	  "tooltip": {
		"trigger": 'axis',
		"axisPointer": {
		  "type": 'shadow'
		}
	  },
	  "legend": {
		"position": "right"
	#     "data": ['Profit', 'Expenses', 'Income']
	  },
	  "grid": {
		"left": '3%',
		"right": '4%',
		"bottom": '3%',
		"containLabel": True
	  },
	  "yAxis": [
		{
		  "type": 'value',
			"show": False
		}
	  ],
	  "xAxis": [
		{
		  "type": 'category',
		  "axisTick": {
			"show": False
		  },
		  "data": ["Current: £"+str(int(current_total_energy_cost))+"/yr", "Future: £"+str(int(future_total_energy_cost))+"/yr"]
		}
	  ],
	  "series": [
		{
		  "name": 'Electricity: Standing Charge',
		  "type": 'bar',
		  "stack": 'Cost',      
		  "label": {
			"show": False,
			"position": 'inside'
		  },
		  "emphasis": {
			"focus": 'series'
		  },
		  "data": elec_st_charge
		},
		{
		  "name": 'Electricity: Baseload',
		  "type": 'bar',
		  "stack": 'Cost',      
		  "label": {
			"show": True,
			"position": 'inside'
		  },
		  "emphasis": {
			"focus": 'series'
		  },
		  "data": elec_baseload
		},    
		{
		  "name": 'Electricity: Heat Pump',
		  "type": 'bar',
		  "stack": 'Cost',      
		  "label": {
			"show": True,
			"position": 'inside'
		  },
		  "emphasis": {
			"focus": 'series'
		  },
		  "data": elec_heatpump
		},
		{
		  "name": 'Electricity: EV',
		  "type": 'bar',
		  "stack": 'Cost',      
		  "label": {
			"show": True,
			"position": 'inside'
		  },
		  "emphasis": {
			"focus": 'series'
		  },
		  "data": elec_ev
		},
		{
		  "name": 'Gas: Standing Charge',
		  "type": 'bar',
		  "stack": 'Cost',      
		  "label": {
			"show": False,
			"position": 'inside'
		  },
		  "emphasis": {
			"focus": 'series'
		  },
		  "data": gas_st_charge
		},    
		{
		  "name": 'Gas: Heating Usage',
		  "type": 'bar',
		  "stack": 'Cost',      
		  "label": {
			"show": True,
			"position": 'inside'
		  },
		  "emphasis": {
			"focus": 'series'
		  },
		  "data": gas_usage
		} ,
		{
		  "name": 'Petrol/Diesel',
		  "type": 'bar',
		  "stack": 'Cost',      
		  "label": {
			"show": True,
			"position": 'inside'
		  },
		  "emphasis": {
			"focus": 'series'
		  },
		  "data": ice_fuel
		} ,    
		{
		  "name": 'Electricity: Export',
		  "type": 'bar',
		  "stack": 'Cost',      
		  "label": {
			"show": True,
			"position": 'inside'
		  },
		  "emphasis": {
			"focus": 'series'
		  },
		  "data": elec_export
		}    
	  ]
	}

	st.subheader("Electricity Usage and Cost")
	
	elec_demand_options = {
	  "tooltip": {
		"trigger": 'axis',
		"axisPointer": {
		  "type": 'shadow'
		}
	  },
	  "legend": {
	    "data": ['Household Appliances', 'Heat Pump', 'Electric Vehicle']
	  },
	  "grid": {
		"left": '3%',
		"right": '4%',
		"bottom": '3%',
		"containLabel": True
	  },
	  "yAxis": [
		{
		  "type": 'value',
			"show": False
		}
	  ],
	  "xAxis": [
		{
		  "type": 'category',
		  "axisTick": {
			"show": False
		  },
		  "data": ['Current Demand (kWh)', 'Future Demand (kWh)']
		}
	  ],
	  "series": [
		{
		  "name": 'Household Appliances',
		  "type": 'bar',
		  "stack": 'Total',      
		  "label": {
			"show": True,
			"position": 'inside'
		  },
		  "emphasis": {
			"focus": 'series'
		  },
		  "data": [round(current_baseload_demand_Wh/1000.,0), round(future_baseload_demand_Wh/1000.,0)]
		},
		{
		  "name": 'Heat Pump',
		  "type": 'bar',
		  "stack": 'Total',      
		  "label": {
			"show": True,
			"position": 'inside'
		  },
		  "emphasis": {
			"focus": 'series'
		  },
		  "data": [round(current_heatpump_demand_Wh/1000.,0), round(future_heatpump_demand_Wh/1000.,0)]
		},
		{
		  "name": 'Electric Vehicle',
		  "type": 'bar',
		  "stack": 'Total',      
		  "label": {
			"show": True,
			"position": 'inside'
		  },
		  "emphasis": {
			"focus": 'series'
		  },
		  "data": [round(current_ev_demand_Wh/1000.,0), round(future_ev_demand_Wh/1000.,0)]
		}
	  ]
	}

	col1, col3, col2 = st.columns([12,1,10])	
	with col1:
		st_echarts(options=elec_demand_options,
					height="200px",)
		
	products_used = list(changes_df['Type'].values)
	
	
	with col2:
		if future_total_demand_Wh > current_total_demand_Wh:
			st.write("You'll be using more electricity overall (excluding batteries)- ",int(round(future_total_demand_Wh/1000.,0)),"kWh, vs. ",int(round(current_total_demand_Wh/1000.,0)),"kWh today.")
		elif future_total_demand_Wh == current_total_demand_Wh:
			st.write("You'll be using the same amount of electricity overall (excluding batteries) - ",int(round(future_total_demand_Wh/1000.,0)),"kWh per year")
		if future_solar_pv_generation_Wh > 0:
			st.write("With Solar PV, you'll be generating ",int(future_solar_pv_generation_Wh/1000.),"kWh per year, powering household appliances during the day and charging the battery (if installed).")
	
	st.markdown("""---""")		
	
	st.subheader('Detailed Analysis')
	
	display_scenario_type = st.selectbox('Scenario to Display', options=['Future','Current'])

	st.subheader('Demand & Generation Profile - Typical Day')

	if display_scenario_type == 'Current':
		selected_scenario_id = current_scenario_id

	if display_scenario_type == 'Future':
		selected_scenario_id = future_scenario_id	




	typical_demand_profile_df = evaluate_scenario(df, scenarios_dict[selected_scenario_id], selected_scenario_id).groupby(['time_of_day']).agg({'grid_elec_import_Wh':'mean', 
																																'electricity_import_unit_rate_per_kWh':'mean',
																																'electricity_export_unit_rate_per_kWh':'mean',
																																'battery_charging_demand_Wh':'mean',
																																'battery_generation_Wh':'mean', 
																																'solar_pv_generation_Wh':'mean',
																																'baseload_demand_Wh':'mean',
																																'ev_charging_demand_Wh':'mean',
																																'electricity_demand_heatpump_Wh':'mean',
																																'grid_elec_export_Wh':'mean',}).reset_index()


	typical_demand_profile_df['household_total_demand'] = (typical_demand_profile_df['baseload_demand_Wh'] + 
																 typical_demand_profile_df['electricity_demand_heatpump_Wh'] + 
																 typical_demand_profile_df['ev_charging_demand_Wh'] +
																 typical_demand_profile_df['battery_charging_demand_Wh']
																 )

	typical_demand_profile_df['solar_pv_exported_Wh'] = typical_demand_profile_df['solar_pv_generation_Wh'] - typical_demand_profile_df['household_total_demand']

	typical_demand_profile_df['battery_net_balance_Wh'] = (typical_demand_profile_df['battery_charging_demand_Wh']-typical_demand_profile_df['battery_generation_Wh'])	

	fraction_demand_off_peak = typical_demand_profile_df.loc[typical_demand_profile_df['electricity_import_unit_rate_per_kWh']<typical_demand_profile_df['electricity_import_unit_rate_per_kWh'].median()]['grid_elec_import_Wh'].sum() / typical_demand_profile_df['grid_elec_import_Wh'].sum()


	option = {
		"tooltip": {"trigger": "axis"},
		"legend": {"data": ['Household Appliances','Heat Pump','EV','Solar PV Generation'],
					"x":"right",
					"y":"bottom"},
		"grid": {"left": "3%", "right": "4%", "bottom": "10%", "containLabel": True},
		"xAxis": {
			"type": "category",
			"data": list(typical_demand_profile_df['time_of_day'].values),
		},
		"yAxis": [{"type": "value", "position": "left"},
				  {"type": "value", "position": "right", "show":False}
					],
		"series": [
			{"name":'Household Appliances',"data": list(typical_demand_profile_df['baseload_demand_Wh'].values/1000.), "type": "bar", "stack":"total","yAxisIndex": 0},
			{"name":'Heat Pump',"data": list(typical_demand_profile_df['electricity_demand_heatpump_Wh'].values/1000.), "type": "bar", "stack":"total","yAxisIndex": 0},
			{"name":'EV',"data": list(typical_demand_profile_df['ev_charging_demand_Wh'].values/1000.), "type": "bar", "stack":"total","yAxisIndex": 0},
			{"name":'Solar PV Generation',"data": list(typical_demand_profile_df['solar_pv_generation_Wh'].values/1000.), "type": "line", "yAxisIndex": 0},			
			],	
	}	
	st_echarts(
		options=option, 
		height="300px",
	)	
	
	st.subheader('Grid Import - Typical Day')



	option = {
		"tooltip": {"trigger": "axis"},
		"legend": {"data": ["Elec Imported (kWh)","Import Price (£/kWh)"],
					"x":"right",
					"y":"bottom"},
		"grid": {"left": "3%", "right": "4%", "bottom": "10%", "containLabel": True},
		"xAxis": {
			"type": "category",
			"data": list(typical_demand_profile_df['time_of_day'].values),
		},
		"yAxis": [{"type": "value", "position": "left", },
					{"type": "value", "position": "right", "show":False}
					],
		"series": [
			{"name":'Elec Imported (kWh)',"data": list(typical_demand_profile_df['grid_elec_import_Wh'].values/1000.), "type": "bar", "yAxisIndex": 0},
			{"name":'Import Price (£/kWh)',"data": list(typical_demand_profile_df['electricity_import_unit_rate_per_kWh'].values), "type": "line", "yAxisIndex": 1},
			],	
	}	
	col1, col3, col2 = st.columns([10,1,5])

	with col1:
		st_echarts(
			options=option, 
			height="300px",
		)
	with col2:

		if display_scenario_type == 'Future':

			st.write(int(future_grid_import_Wh/1000.),'kWh of electricity is imported from the grid every year, costing you £',int(future_grid_elec_import_cost))
			st.write(int(100.*fraction_demand_off_peak),'% of grid import is during off-peak hours')
			st.write('This is due to battery storage and EV preferentially charging at cheaper electricity import rates')
			if future_elec_effective_cost_per_kWh <= current_elec_effective_cost_per_kWh:
				st.write("Average cost per unit of electricity imported will drop to £",round(future_elec_effective_cost_per_kWh,2),"/kWh from £",round(current_elec_effective_cost_per_kWh,2),'/kWh')
			else:
				st.write("Average cost per unit of electricity imported will increase to £",round(future_elec_effective_cost_per_kWh,2),"/kWh from £",round(current_elec_effective_cost_per_kWh,2),'/kWh')			

		if display_scenario_type == 'Current':

			st.write(int(current_total_demand_Wh/1000.),'kWh of electricity is imported from the grid every year, costing you £',int(current_grid_elec_import_cost))
			st.write("Average cost per unit of electricity is currently: £",round(current_elec_effective_cost_per_kWh,4),'/kWh')

	st.markdown("""---""")

	st.subheader('Grid Export - Typical Day')

	option = {
		"tooltip": {"trigger": "axis"},
		"legend": {"data": ['Grid Export (kWh)','Export Price (£/kWh)'],
					"x":"right",
					"y":"bottom"},
		"grid": {"left": "3%", "right": "4%", "bottom": "10%", "containLabel": True},
		"xAxis": {
			"type": "category",
			"data": list(typical_demand_profile_df['time_of_day'].values),
		},
		"yAxis": [{"type": "value", "position": "left"},
				  {"type": "value", "position": "right", "show":False}
					],
		"series": [
			{"name":'Grid Export (kWh)',"data": list(typical_demand_profile_df['grid_elec_export_Wh'].values/1000.), "type": "bar","yAxisIndex": 0},
			{"name":'Export Price (£/kWh)',"data": list(typical_demand_profile_df['electricity_export_unit_rate_per_kWh'].values), "type": "line", "yAxisIndex": 1},
			],	
	}	
	col1, col3, col2 = st.columns([10,1,5])	
	with col1:
		st_echarts(
			options=option, 
			height="300px",
		)	

	with col2:

		if display_scenario_type == 'Future':

			st.write(int(future_grid_elec_export_Wh/1000.),'kWh of electricity is exported to the grid every year, earning you £',int(future_elec_export_income))
			st.write("On average, every unit of electricity sold back to the grid is worth £",round((1000.*future_elec_export_income / future_grid_elec_export_Wh),3),"/kWh")


		if display_scenario_type == 'Current':
			selected_scenario_id = future_scenario_id	

			st.write(int(current_grid_elec_export_Wh/1000.),'kWh of electricity is exported to the grid every year, earning you £',int(current_elec_export_income))
			if current_grid_elec_export_Wh > 0.:
				st.write("On average, every unit of electricity sold back to the grid is worth £",round((1000.*current_elec_export_income / current_grid_elec_export_Wh),3),"/kWh")
				
		st.write("""Export happens when:
- Excess electricity is generated by solar PV
- Batteries preferentially discharge during peak periods
		""")

	st.markdown("""---""")	

	if ('Battery Storage' in product_type) or (summary_results_df.loc[selected_future_scenario_cond]['battery_storage_capacity_Wh'].values[0] > 0.) or (summary_results_df.loc[current_cond]['battery_storage_capacity_Wh'].values[0] > 0.):


		st.subheader('Battery Charging/Discharging - Typical Day')


		option = {
	# 		"title": {"text": "Battery Charging/Discharging in kWh"},
			"tooltip": {"trigger": "axis"},
			"legend": {"data": ['Battery Charging (kWh)','Battery Discharging (kWh)'],
						"x":"right",
						"y":"bottom"},
			"grid": {"left": "3%", "right": "4%", "bottom": "10%", "containLabel": True},
			"xAxis": {
				"type": "category",
				"data": list(typical_demand_profile_df['time_of_day'].values),
			},
			"yAxis": [{"type": "value", "position": "left"},
	# 					{"type": "value", "position": "right", "show":False}
						],
			"series": [
				{"name":'Battery Charging (kWh)',"data": list(typical_demand_profile_df['battery_charging_demand_Wh'].values/1000.), "type": "bar","stack": 'total',"yAxisIndex": 0},
				{"name":'Battery Discharging (kWh)',"data": list(-typical_demand_profile_df['battery_generation_Wh'].values/1000.), "type": "bar","stack": 'total',"yAxisIndex": 0},			
				],	
		}	

		col1, col3, col2 = st.columns([10,1,5])	
		with col1:

			st_echarts(
				options=option, 
				height="300px",
			)
		with col2:
			st.write("""Batteries charge when:
- Import prices are cheap, during off-peak periods
- Solar PV generates excess electricity, if installed
			""")
			st.write("")
			st.write("""Batteries discharge when:
- Export prices are highest, during peak periods
- To satisfy household demand during non off-peak periods
			""")

def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def color_avail(val):
    color = 'lightgreen' if val=='✓' else 'orange'
    return f'background-color: {color}'

def generate_waterfall_chart(summary_results_df, 
								current_scenario_idx, 
								future_scenario_idx
							 ):

	# We need to time-stepped analysis - 

		current_cond = (summary_results_df['scenario_id'] == current_scenario_idx)
		future_cond = (summary_results_df['scenario_id'] == future_scenario_idx)

		current_elec_standing_charge = summary_results_df.loc[current_cond]['electricity_standing_charge_annual'].values[0]
		current_elec_cost = summary_results_df.loc[current_cond]['electricity_import_cost'].values[0] - summary_results_df.loc[current_cond]['electricity_export_revenue'].values[0] + summary_results_df.loc[current_cond]['electricity_standing_charge_annual'].values[0]
		current_gas_cost = summary_results_df.loc[current_cond]['gas_import_cost'].values[0] + summary_results_df.loc[current_cond]['gas_standing_charge_annual'].values[0]
		current_ice_fuel_cost = summary_results_df.loc[current_cond]['vehicle_fuel_cost'].values[0]
		current_total_energy_cost = (current_elec_cost+current_gas_cost+current_ice_fuel_cost)

		future_elec_standing_charge = summary_results_df.loc[future_cond]['electricity_standing_charge_annual'].values[0]

		future_elec_cost = (summary_results_df.loc[future_cond]['electricity_import_cost'].values[0] 
							- summary_results_df.loc[future_cond]['electricity_export_revenue'].values[0]
							+ summary_results_df.loc[future_cond]['electricity_standing_charge_annual'].values[0])
						
		future_gas_cost = (summary_results_df.loc[future_cond]['gas_import_cost'].values[0]
						   + summary_results_df.loc[future_cond]['gas_standing_charge_annual'].values[0])
					   
		future_ice_fuel_cost = summary_results_df.loc[future_cond]['vehicle_fuel_cost'].values[0]

		future_total_energy_cost = (future_elec_cost+future_gas_cost+future_ice_fuel_cost)

		diff_elec_cost = future_elec_cost - current_elec_cost
		diff_gas_cost = future_gas_cost - current_gas_cost
		diff_ice_fuel_cost = future_ice_fuel_cost - current_ice_fuel_cost

		current_elec_export_income = summary_results_df.loc[current_cond]['electricity_export_revenue'].values[0]

		fig = go.Figure(go.Waterfall(
	        name = "",
	        measure = ["absolute", "relative", "relative", "relative", "total"],
	        x = ["Current Bill", "Electricity", "Gas", "Petrol/Diesel", "Future Bill"],
	        textposition = "outside",
	        # text = ["+60", "+80", "", "-40", "-20", "Total"],
	        y = [current_total_energy_cost, 
	        	 diff_elec_cost, 
	        	 diff_gas_cost,
	        	 diff_ice_fuel_cost, 
	        	 future_total_energy_cost],
	        connector = {"line":{"color":"rgb(63, 63, 63)"}},
	    ))

		fig.update_layout(
	            title = "Bill Changes",
	            showlegend = False
	    )


		
		st.plotly_chart(fig, theme="streamlit")
		
if __name__ == '__main__':

	vehicles_df, solar_pv_systems_df, heating_systems_df, battery_storage_systems_df, \
	ev_chargers_df, energy_tariffs_df, locations_df , installers_df, postcodes_df = load_common_inputs()
	cols = ['heating_system_name','solar_pv_name','battery_storage_name','vehicle_name','ev_charger_name','tariff_name']
	litres_per_gallon = 4.546
	
	# Location

	if 'property_details_set' not in st.session_state:
		set_property_details()

	if st.sidebar.button('Edit Property Details', type='secondary'):
		set_property_details()

	postcode_outcode = st.session_state.postcode[:-3].strip().upper()
	postcode_region = postcodes_df.loc[postcodes_df['postcode']==postcode_outcode]['uk_region'].values[0]
	
	annual_electricity_consumption_kWh = st.session_state.annual_electricity_consumption_kWh
	annual_user_gas_demand_kWh = st.session_state.annual_user_gas_demand_kWh
	latitude = st.session_state.latitude
	longitude = st.session_state.longitude

	location_path = 'pvgis_inputs/'
	pvgis_files = [f for f in listdir(location_path) if isfile(join(location_path, f)) and fnmatch.fnmatch(f, '*.json')]
	location_names = list([n.split('_')[0] for n in pvgis_files])

	if 'technology_choices' not in st.session_state:
		set_upgrades(vehicles_df, solar_pv_systems_df, heating_systems_df,
							battery_storage_systems_df, ev_chargers_df, energy_tariffs_df,
							locations_df , installers_df)

	if st.sidebar.button('Edit Product Upgrades', type='secondary'):
		set_upgrades(vehicles_df, solar_pv_systems_df, heating_systems_df,
							battery_storage_systems_df, ev_chargers_df, energy_tariffs_df,
							locations_df , installers_df)

	current_battery_storage_system, current_battery_num_units,\
	battery_storage_option, battery_number_units, \
	current_heating_system, future_heating_system, \
	current_solar_PV_Wp, current_solar_pv_system, \
	future_solar_pv_power_Wp, solar_pv_option, azimuth, tilt,\
	current_energy_tariff, energy_tariff_option, current_vehicle, future_vehicle,\
	home_departure_hh_period, home_arrival_hh_period, arrival_departure_delta_n_hh_periods, \
	annual_miles_driven, export_limit_kW, vehicle_fuel_cost_per_litre= map(st.session_state.technology_choices.get,('current_battery_storage_system',
		'current_battery_num_units',
		'battery_storage_option',
		'battery_number_units',
		'current_heating_system',
		'future_heating_system',
		'current_solar_PV_Wp',
		'current_solar_pv_system',
		'future_solar_pv_power_Wp',
		'solar_pv_option',
		'azimuth',
		'tilt',
		'current_energy_tariff',
		'energy_tariff_option',
		'current_vehicle',
		'future_vehicle',
		'home_departure_hh_period',
		'home_arrival_hh_period',
		'arrival_departure_delta_n_hh_periods',
		'annual_miles_driven',
		'export_limit_kW',
		'vehicle_fuel_cost_per_litre'
		))

	st.title('Cut My Energy Bill')
	st.write('_See what your low carbon technology setup could save you in bills, without speaking to a single salesperson_')
	col1, col2 = st.columns([4,1],gap='small')

	# Generate the half-hourly df
	profile_name = 'ProfileClass1'


	energy_tariffs_df, rates_df_pivoted = expand_energy_tariffs(energy_tariffs_df, home_arrival_hh_period)
	half_hourly_df = generate_half_hourly_electricity_baseload(profile_name, annual_electricity_consumption_kWh, home_arrival_hh_period)

# 	Getting solar PV potential, temperature, from PVGIS
	df = get_hourly_PVGIS_file(latitude, longitude, azimuth, tilt)

	df = calculate_heat_demand(df, annual_user_gas_demand_kWh)

	# after calculating the modelled heat demand, merge df with half-hourly-df
	df = pd.merge(df,half_hourly_df[['datetime','day_of_week','hh_period','electricity_demand_modelled_Wh','periods_since_monday_home_arrival']],on='datetime')

	ev_Wh_per_mile=250.
	ev_max_power_W = 7000.

# 	battery_number_units = np.arange(battery_min_number_units, battery_max_number_units+1, 1)    
	battery_units_df = pd.DataFrame(data={'battery_num_units':[battery_number_units]})

	ev_demand_dict_list = calculate_EV_charging_behaviour(rates_df_pivoted, annual_miles_driven, ev_Wh_per_mile, ev_max_power_W, arrival_departure_delta_n_hh_periods)

	# Feed in the user selections to the scenarios creation step here instead...

	scenario_df, scenarios_dict = create_scenarios(vehicles_df, battery_storage_systems_df, battery_units_df,
									heating_systems_df, solar_pv_systems_df, 
									# solar_power_df,
						 			ev_chargers_df, energy_tariffs_df, ev_demand_dict_list,
						 			st.session_state.technology_choices)
						 
	if 'results_present' not in st.session_state:
		set_results_require_rerun()


	if st.sidebar.button('Show Results', type='primary'):
		with st.spinner(text='Running '+str(len(scenario_df))+' scenarios - this may take a few minutes...'):

			st.session_state.summary_results_df = process_scenarios(df, scenario_df, scenarios_dict)
			st.session_state.results_present = True

	if not st.session_state.results_present:
		st.caption('Inputs Updated - Refresh Results with Button Above')		
	
	else:

		summary_results_df = st.session_state.summary_results_df
		
		summary_results_df = pd.merge(summary_results_df, scenario_df, on='scenario_id')
	
		summary_results_df['solar_pv_generation_exported_Wh'] = (summary_results_df['solar_pv_generation_Wh'] - 
																 summary_results_df['baseload_demand_Wh'] - 
																 summary_results_df['electricity_demand_heatpump_Wh'] - 
																 summary_results_df['ev_charging_demand_Wh']
																 )

		summary_results_df['solar_pv_generation_exported_fraction'] = np.minimum(0.,summary_results_df['solar_pv_generation_exported_Wh'])
	
		summary_results_df['vehicle_fuel_cost'] = 0.

		gasoline_veh_cond = (summary_results_df['vehicle_fuel_type'] == 'gasoline')

		summary_results_df.loc[gasoline_veh_cond,'vehicle_fuel_cost'] = (annual_miles_driven * 
																		  vehicle_fuel_cost_per_litre * 
																		  (litres_per_gallon / summary_results_df['vehicle_miles_per_gallon'].loc[gasoline_veh_cond]))


		summary_results_df['total_energy_cost'] = (summary_results_df['electricity_import_cost']+
												   summary_results_df['gas_import_cost']+
												   summary_results_df['vehicle_fuel_cost']-
												   summary_results_df['electricity_export_revenue']
												  )

		summary_results_df.sort_values(by='total_energy_cost', ascending=True, inplace=True)

		summary_results_df['electricity_standing_charge_annual'] = (summary_results_df['electricity_standing_charge_daily']*365.).round(2)
		summary_results_df['gas_standing_charge_annual'] = (summary_results_df['gas_standing_charge_daily']*365.).round(2)

		no_gas_cond = (summary_results_df['heating_system_fuel_type']!='gas')
		summary_results_df.loc[no_gas_cond, 'gas_standing_charge_annual'] = 0.
	
	

		summary_results_df['vehicle_fuel_cost'] = summary_results_df['vehicle_fuel_cost'].round(2)
		summary_results_df['electricity_import_cost'] = summary_results_df['electricity_import_cost'].round(2)
		summary_results_df['gas_import_cost'] = summary_results_df['gas_import_cost'].round(2)
		summary_results_df['electricity_export_revenue'] = summary_results_df['electricity_export_revenue'].round(2)

		summary_results_df['solar_pv_cost'] = summary_results_df['solar_pv_cost_per_kWp'] * summary_results_df['solar_pv_power_kWp']

		# Missing here are the standing charges!
		summary_results_df['Annual Cost'] = (summary_results_df['vehicle_fuel_cost']+
													summary_results_df['electricity_import_cost']+
													summary_results_df['gas_import_cost']-
													summary_results_df['electricity_export_revenue']+
													summary_results_df['electricity_standing_charge_annual']+
													summary_results_df['gas_standing_charge_annual']
													).round(2)


		summary_results_df.sort_values(by='Annual Cost', ascending=True, inplace=True)
		
		current_cond = (
				(summary_results_df['heating_system_name']==current_heating_system) &
				(summary_results_df['vehicle_name'] == current_vehicle)&
				(summary_results_df['battery_storage_name']==current_battery_storage_system)&
				(summary_results_df['solar_pv_name']==current_solar_pv_system)&
				(summary_results_df['tariff_name']==current_energy_tariff)
				)

		summary_results_df['num_upgrades'] = 0
		summary_results_df.loc[summary_results_df['heating_system_name']!=current_heating_system, 'num_upgrades'] += 1	
		summary_results_df.loc[summary_results_df['battery_storage_name']!=current_battery_storage_system, 'num_upgrades'] += 1
		summary_results_df.loc[summary_results_df['solar_pv_name']!=current_solar_pv_system, 'num_upgrades'] += 1
		summary_results_df.loc[summary_results_df['tariff_name']!=current_energy_tariff, 'num_upgrades'] += 1
		summary_results_df.loc[summary_results_df['vehicle_name']!=current_vehicle, 'num_upgrades'] += 1
		
		summary_results_df['Heating'] = ''
		summary_results_df.loc[summary_results_df['heating_system_name']!=current_heating_system,'Heating'] = '✓'

		summary_results_df['Battery'] = ''
		summary_results_df.loc[summary_results_df['battery_storage_name']!=current_battery_storage_system,'Battery'] = '✓'

		summary_results_df['Solar PV'] = ''
		summary_results_df.loc[summary_results_df['solar_pv_name']!=current_solar_pv_system,'Solar PV'] = '✓'

		summary_results_df['Tariff'] = ''
		summary_results_df.loc[summary_results_df['tariff_name']!=current_energy_tariff,'Tariff'] = '✓'

		# summary_results_df['EV Charger'] = ''
		# summary_results_df['EV Charger'].loc[summary_results_df['ev_charger_name']!=current_ev_charger] = '✓'

		summary_results_df['Vehicle'] = ''
		summary_results_df.loc[summary_results_df['vehicle_name']!=current_vehicle,'Vehicle'] = '✓'
		current_elec_standing_charge = summary_results_df.loc[current_cond]['electricity_standing_charge_annual'].values[0]
		current_elec_cost = summary_results_df.loc[current_cond]['electricity_import_cost'].values[0] - summary_results_df.loc[current_cond]['electricity_export_revenue'].values[0] + summary_results_df.loc[current_cond]['electricity_standing_charge_annual'].values[0]
		current_gas_cost = summary_results_df.loc[current_cond]['gas_import_cost'].values[0] + summary_results_df.loc[current_cond]['gas_standing_charge_annual'].values[0]
		current_ice_fuel_cost = summary_results_df.loc[current_cond]['vehicle_fuel_cost'].values[0]
		current_total_energy_cost = (current_elec_cost+current_gas_cost+current_ice_fuel_cost)

		current_elec_effective_cost_per_kWh = summary_results_df.loc[current_cond]['electricity_import_cost'].values[0]/(summary_results_df.loc[current_cond]['grid_elec_import_Wh'].values[0]*0.001)


		current_baseload_demand_Wh = annual_electricity_consumption_kWh*1000.
		current_ev_demand_Wh = summary_results_df.loc[current_cond]['ev_charging_demand_Wh'].values[0]
		current_heatpump_demand_Wh = summary_results_df.loc[current_cond]['electricity_demand_heatpump_Wh'].values[0]
		current_solar_pv_generation_Wh = summary_results_df.loc[current_cond]['solar_pv_generation_Wh'].values[0]
		current_grid_elec_export_Wh = summary_results_df.loc[current_cond]['grid_elec_export_Wh'].values[0]
		current_solar_pv_self_consumed_Wh = current_solar_pv_generation_Wh - current_grid_elec_export_Wh
		current_total_demand_Wh = current_baseload_demand_Wh + current_ev_demand_Wh + current_heatpump_demand_Wh

		current_grid_elec_import_cost = summary_results_df.loc[current_cond]['electricity_import_cost'].values[0]
		current_baseload_elec_cost = current_grid_elec_import_cost * (current_baseload_demand_Wh/(current_baseload_demand_Wh+current_ev_demand_Wh+current_heatpump_demand_Wh))
		current_ev_elec_cost = current_grid_elec_import_cost * (current_ev_demand_Wh/(current_baseload_demand_Wh+current_ev_demand_Wh+current_heatpump_demand_Wh))
		current_heatpump_elec_cost = current_grid_elec_import_cost * (current_heatpump_demand_Wh/(current_baseload_demand_Wh+current_ev_demand_Wh+current_heatpump_demand_Wh))
		current_solar_pv_export_income = summary_results_df.loc[current_cond]['electricity_export_revenue'].values[0]
		current_gas_standing_charge = summary_results_df.loc[current_cond]['gas_standing_charge_annual'].values[0]
		current_gas_usage_cost = summary_results_df.loc[current_cond]['gas_import_cost'].values[0]

		summary_results_df['Annual Savings'] = (current_total_energy_cost-
														summary_results_df['Annual Cost']).round(0)

		summary_results_df['total_system_cost'] = (summary_results_df['heating_system_cost']+
													summary_results_df['solar_pv_cost']+
													summary_results_df['battery_storage_cost']+
													summary_results_df['ev_charger_cost']
													)
		

		current_scenario_idx = summary_results_df.loc[current_cond]['scenario_id'].values[0]
		print (current_scenario_idx)
		
		current_scenario_investment = summary_results_df.loc[current_cond]['total_system_cost'].values[0]
		

		# If the current system is the same as the future system, then no upgrade is required for that technology
		# and therefore the cost should be removed.


		print (current_scenario_investment)

		summary_results_df['Upgrade Cost'] = (summary_results_df['total_system_cost']-
												current_scenario_investment
												)

		print (summary_results_df[['scenario_id','total_system_cost','heating_system_cost','solar_pv_cost','battery_storage_cost','ev_charger_cost','Upgrade Cost']])


		heating_change_cond = (summary_results_df['heating_system_name'] != current_heating_system)

		summary_results_df.loc[heating_change_cond,'Upgrade Cost'] += summary_results_df.loc[current_cond]['heating_system_cost'].values[0]

		summary_results_df['10 Year Return'] = ((summary_results_df['Annual Savings']*10) - 
												summary_results_df['Upgrade Cost'])
											
		summary_results_df['25 Year Return'] = ((summary_results_df['Annual Savings']*25) - 
												summary_results_df['Upgrade Cost'])											
	
	
		summary_results_df['payback'] = summary_results_df['Upgrade Cost'] / summary_results_df['Annual Savings']
	
		summary_results_df['Payback (Years)'] = np.ceil(summary_results_df['Upgrade Cost'] / summary_results_df['Annual Savings'])

		future_potential_cond = (
				(summary_results_df['heating_system_name'] == future_heating_system) &
				(summary_results_df['vehicle_name'] == future_vehicle)&
				(summary_results_df['battery_storage_name'] == battery_storage_option)&
				(summary_results_df['solar_pv_name'] == solar_pv_option)&
				(summary_results_df['tariff_name'] == energy_tariff_option)
				)
		
		largest_annual_savings_idx = summary_results_df.loc[future_potential_cond].sort_values(by='Annual Cost',ascending=True)['scenario_id'].values[0]
		best_return_10_years_idx = summary_results_df.sort_values(by='10 Year Return',ascending=False)['scenario_id'].values[0]	
		best_return_25_years_idx = summary_results_df.sort_values(by='25 Year Return',ascending=False)['scenario_id'].values[0]	
		fastest_payback_idx = summary_results_df.loc[summary_results_df['payback']>0.].sort_values(by='payback',ascending=True)['scenario_id'].values[0]	



		largest_annual_savings_cond =(summary_results_df['scenario_id']==largest_annual_savings_idx)
		best_return_10_years_cond = (summary_results_df['scenario_id']==best_return_10_years_idx)
		best_return_25_years_cond = (summary_results_df['scenario_id']==best_return_25_years_idx)
		fastest_payback_cond = (summary_results_df['scenario_id']==fastest_payback_idx)

		n_scenarios = len(summary_results_df.loc[future_potential_cond].index)

		selected_scenario_id = largest_annual_savings_idx
		selected_scenario_cond = largest_annual_savings_cond

		old_annual_bill = int(summary_results_df.loc[current_cond]['Annual Cost'].values[0])
		new_annual_bill = int(summary_results_df.loc[future_potential_cond].loc[selected_scenario_cond]['Annual Cost'].values[0])
		bill_difference = old_annual_bill - new_annual_bill

		with st.container(border=True):
			msg = """**Your Chosen Option:**
			"""
			st.write(msg)	
			col1, col2, col3, col4 = st.columns([1,1,1,1],gap='small')		
			with col1:
				# Show annual savings
				if bill_difference <= 0.:
					st.metric(label=":red[Annual Loss]", value='£'+str(bill_difference))
				
				else:
					st.metric(label=":green[Annual Savings]", value='£'+str(bill_difference))				

			with col2:
				# Show upfront cost needed
				st.metric(label="Upfront Cost", value='£'+str(int(summary_results_df.loc[selected_scenario_cond]['Upgrade Cost'].values[0])),			
						  help="What it'll cost to upgrade to the new system")
				

			with col3:				
				# Show Payback period
				st.metric(label="Payback", value=str(int(summary_results_df.loc[selected_scenario_cond]['Payback (Years)'].values[0]))+' years')
				

			with col4:
				if st.button("""Find an Installer
					""",key='preferred_option',type="primary"):
					show_next_steps(installers_df, postcode_region)
				if st.button("See More Details", type="secondary"):
					# show_details()
					generate_detailed_analysis(current_scenario_idx, selected_scenario_id)

		st.write("***")
		st.write("Other Options Are Available:")
		if best_return_25_years_idx != selected_scenario_id:
			with st.container(border=True):
				msg = """*Best Return Over 25 Years:*
				"""
				st.write(msg)
				# old_annual_bill = int(summary_results_df.loc[current_cond]['Annual Cost'].values[0])
				new_annual_bill = int(summary_results_df.loc[best_return_25_years_cond]['Annual Cost'].values[0])
				bill_difference = old_annual_bill - new_annual_bill
				col1, col2, col3, col4 = st.columns([1,1,1,1],gap='small')		
				with col1:
					# st.metric(label=":red[Old Annual Bill]", value='£'+str(int(summary_results_df.loc[current_cond]['Annual Cost'].values[0])))
					if bill_difference <= 0.:
						st.metric(label=":red[Annual Loss]", value='£'+str(bill_difference))
					
					else:
						st.metric(label=":green[Annual Savings]", value='£'+str(bill_difference))				

				with col2:

					st.metric(label="Upfront Cost", value='£'+str(int(summary_results_df.loc[best_return_25_years_cond]['Upgrade Cost'].values[0])),			
							  help="What it'll cost to upgrade to the new system")


				with col3:
					
					st.metric(label="Payback", value=str(int(summary_results_df.loc[best_return_25_years_cond]['Payback (Years)'].values[0]))+' years')

				with col4:
					if st.button("Find an Installer",key='best_return_25_years',
									type="primary"):
						show_next_steps(installers_df, postcode_region)
					if st.button("See More Details", type="secondary",
									key='best_return_25_years_details'):
						generate_detailed_analysis(current_scenario_idx, best_return_25_years_idx)

		if fastest_payback_idx != selected_scenario_id:
			with st.container(border=True):
				msg = """*Fastest Payback:*
				"""
				st.write(msg)
				# old_annual_bill = int(summary_results_df.loc[current_cond]['Annual Cost'].values[0])
				new_annual_bill = int(summary_results_df.loc[fastest_payback_cond]['Annual Cost'].values[0])
				bill_difference = old_annual_bill - new_annual_bill
				col1, col2, col3, col4 = st.columns([1,1,1,1],gap='small')		
				with col1:
					if bill_difference <= 0.:
						st.metric(label=":red[Annual Loss]", value='£'+str(bill_difference))
					
					else:
						st.metric(label=":green[Annual Savings]", value='£'+str(bill_difference))				
					
				with col2:
					st.metric(label="Upfront Cost", value='£'+str(int(summary_results_df.loc[fastest_payback_cond]['Upgrade Cost'].values[0])),			
							  help="What it'll cost to upgrade to the new system")
	

				with col3:
					
					st.metric(label="Payback", value=str(int(summary_results_df.loc[fastest_payback_cond]['Payback (Years)'].values[0]))+' years')

				with col4:
					if st.button("Find an Installer",key='fastest_payback',type="primary"):
						show_next_steps(installers_df, postcode_region)						
					if st.button("See More Details", type="secondary",key='fastest_payback_details'):
						generate_detailed_analysis(current_scenario_idx, fastest_payback_idx)


# 		Create an Excel file and let users download the full analysis for themselves!
		st.write("***")

		current_half_hourly_results_df = evaluate_scenario(df, scenarios_dict[current_scenario_idx], current_scenario_idx)
		future_half_hourly_results_df = evaluate_scenario(df, scenarios_dict[selected_scenario_id], selected_scenario_id)

		buffer = io.BytesIO()

		df_list = [current_half_hourly_results_df, future_half_hourly_results_df]
		with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
		    # Write each dataframe to a different worksheet.

		    current_half_hourly_results_df.to_excel(writer, sheet_name=f'Current Setup', index=False)
		    future_half_hourly_results_df.to_excel(writer, sheet_name=f'Future Setup', index=False)

		    # Close the Pandas Excel writer and output the Excel file to the buffer
		    writer.close()

		    st.download_button(
		        label="Download Excel Spreadsheet",
		        data=buffer,
		        file_name="cutmyenergybill-export.xlsx",
		        mime="application/vnd.ms-excel"
		    )




	
    
