import streamlit as st
import numpy as np
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder
from streamlit_echarts import st_echarts


st.set_page_config(layout="wide")

summary_results_df = pd.read_csv('summary_results.csv')



cols = ['heating_system_name','solar_pv_name','battery_storage_name','vehicle_name','ev_charger_name','tariff_name']
# df['combined'] = df[cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
summary_results_df['Products'] = summary_results_df[cols].apply(lambda row: ' | '.join(row.values.astype(str)), axis=1)

st.sidebar.title('Domestic Energy Bill Reduction Application (DEBRA)')

with st.sidebar.expander("My Region & Energy Usage"):
	location_name = st.selectbox(
		 'Current Location',
		 np.sort(summary_results_df['location_name'].unique()),
		 help='To calculate temperature & Solar PV'
		 )

	annual_electricity_consumption_kWh = np.sort(summary_results_df['annual_electricity_consumption_kWh'].unique())
	elec_kWh_slider = st.select_slider(
		 'Annual Electricity Consumption (kWh):',
		 options=annual_electricity_consumption_kWh,
		 value=2900,
		 help='Excluding Heat Pumps, EVs, Battery Storage - Defaults to UK avg'
		 )

	annual_gas_consumption_kWh = np.sort(summary_results_df['annual_gas_consumption_kWh'].unique())
	gas_kWh_slider = st.select_slider(
		 'Annual Gas Consumption (kWh):',
		 options=annual_gas_consumption_kWh,
		 value=12000,
		 help='Defaults to UK avg'
		 )

	daily_miles_driven = st.select_slider(
		 'Select average miles driven per day:',
		 options=[20,50,100],
		 value=20,
		 help='Defaults to UK avg'
		 )

with st.sidebar.expander("Currently Installed"):

	current_heating_system = st.selectbox(
		 'Current Heating System',
		 summary_results_df['heating_system_name'].unique()
		 )

	current_solar_pv_system = st.selectbox(
		 'Current Solar PV system',
		 summary_results_df['solar_pv_name'].unique()
		 )

	current_battery_storage_system = st.selectbox(
		 'Current Battery Storage',
		 summary_results_df['battery_storage_name'].unique()
		 )

	current_vehicle = st.selectbox(
		 'Current Vehicle',
		 summary_results_df['vehicle_name'].unique()
		 )

	current_ev_charger = st.selectbox(
		 'Current EV Charger',
		 summary_results_df['ev_charger_name'].unique()
		 )
		 
	current_energy_tariff = st.selectbox(
		 'Current Energy Tariff',
		 summary_results_df['tariff_name'].unique()
		 )


with st.sidebar.expander("Technologies to Consider"):

	future_heating_system = st.multiselect(
				'Future Heating Systems:',
				summary_results_df['heating_system_name'].unique(),
				summary_results_df['heating_system_name'].unique()
# 				['Average Gas Boiler','Air Source Heat Pump'],
# 		 		['Average Gas Boiler','Air Source Heat Pump']
		 		 )


	future_solar_pv_system = st.selectbox(
		 'Solar PV?',
		 ('Yes','No')
		 )
	if future_solar_pv_system == 'Yes':
		solar_pv_size = st.slider('Solar PV Size (kW)', 1.0, 10.0, 4.5, step=0.5, 
				help='Rated Power in kiloWatts, Assume South-facing @ 35Deg Slope')
		
		solar_pv_option = summary_results_df['solar_pv_name'].unique()
	else:
		solar_pv_size = 0.
		solar_pv_option = ['No Solar PV']


	future_battery_storage_system = st.selectbox(
		 'Battery Storage?',
		 ('Yes','No')
		 )
	if future_battery_storage_system == 'No':
		battery_storage_option = ['No Battery Storage']
	else:
		# battery_storage_option = summary_results_df['battery_storage_name'].unique()
		battery_storage_option = st.multiselect(
					'Future Battery Storage',
					summary_results_df['battery_storage_name'].unique(),
					summary_results_df['battery_storage_name'].unique())

		
	electric_vehicle_option = st.selectbox(
		 'Electric Vehicle?',
		 ('Yes','No')
		 )

	if electric_vehicle_option == 'No':
		future_vehicle = ['Typical Petrol Car']
	else:
		# battery_storage_option = summary_results_df['battery_storage_name'].unique()
		future_vehicle = st.multiselect(
					'Future Vehicle',
					summary_results_df['vehicle_name'].unique(),
					summary_results_df['vehicle_name'].unique())		

	future_ev_charger = st.selectbox(
		 'EV Charger?',
		 ('Yes','No')
		 )
	if future_ev_charger == 'No':
		ev_charger_option = ['Standard 3-Pin Home Socket']
	else:
# 		ev_charger_option = summary_results_df['ev_charger_name'].unique()
		ev_charger_option = st.multiselect(
					'Future EV Charger',
					summary_results_df['ev_charger_name'].unique(),
					summary_results_df['ev_charger_name'].unique())


	future_smart_meter = st.selectbox(
		 'Smart Meters',
		 ['Yes','No']
		 )
	if future_smart_meter == 'Yes':
		smart_meter_option = [True]
	else:
		smart_meter_option = [False]
		
	energy_tariff_option = st.multiselect(
		 'Future Energy Tariffs',
		 summary_results_df['tariff_name'].unique(),
		 summary_results_df['tariff_name'].unique()
		 )

    
budget = st.sidebar.slider('Budget (£)', 0, 50000, 30000, step=500)
    
vehicle_fuel_cost_per_litre = st.sidebar.slider('Vehicle Fuel Cost (£/litre)', 1.00, 2.50, 1.90, step=0.01)

st.sidebar.write('Kindly supported by [Climate Subak](https://climatesubak.org/)')




summary_results_df['vehicle_fuel_cost'] = (summary_results_df['vehicle_litres_fuel_annual']*vehicle_fuel_cost_per_litre).round(2)

summary_results_df['electricity_standing_charge_annual'] = (summary_results_df['electricity_standing_charge_daily']*365.).round(2)
summary_results_df['gas_standing_charge_annual'] = (summary_results_df['gas_standing_charge_daily']*365.).round(2)

no_gas_cond = (summary_results_df['heating_system_fuel_type']!='gas')
summary_results_df['gas_standing_charge_annual'].loc[no_gas_cond] = 0.
	
	

summary_results_df['vehicle_fuel_cost'] = summary_results_df['vehicle_fuel_cost'].round(2)
summary_results_df['grid_elec_import_cost'] = summary_results_df['grid_elec_import_cost'].round(2)
summary_results_df['gas_import_cost'] = summary_results_df['gas_import_cost'].round(2)
summary_results_df['electricity_export_income'] = summary_results_df['electricity_export_income'].round(2)

# Missing here are the standing charges!
summary_results_df['Annual Cost'] = (summary_results_df['vehicle_fuel_cost']+
											summary_results_df['grid_elec_import_cost']+
											summary_results_df['gas_import_cost']-
											summary_results_df['electricity_export_income']+
											summary_results_df['electricity_standing_charge_annual']+
											summary_results_df['gas_standing_charge_annual']
											).round(2)


summary_results_df.sort_values(by='Annual Cost', ascending=True, inplace=True)
# print (future_vehicle)
current_cond = ((summary_results_df['daily_miles_driven']==daily_miles_driven) &
		(summary_results_df['annual_electricity_consumption_kWh']==elec_kWh_slider) &
		(summary_results_df['annual_gas_consumption_kWh']==gas_kWh_slider) &
		(summary_results_df['heating_system_name']==current_heating_system) &
		(summary_results_df['vehicle_name'] == current_vehicle)&
		(summary_results_df['battery_storage_name']==current_battery_storage_system)&
		(summary_results_df['solar_pv_name']==current_solar_pv_system)&
		(summary_results_df['tariff_name']==current_energy_tariff)&
		(summary_results_df['ev_charger_name']==current_ev_charger)		
		)

current_elec_standing_charge = summary_results_df.loc[current_cond]['electricity_standing_charge_annual'].values[0]
current_elec_cost = summary_results_df.loc[current_cond]['grid_elec_import_cost'].values[0] - summary_results_df.loc[current_cond]['electricity_export_income'].values[0] + summary_results_df.loc[current_cond]['electricity_standing_charge_annual'].values[0]
current_gas_cost = summary_results_df.loc[current_cond]['gas_import_cost'].values[0] + summary_results_df.loc[current_cond]['gas_standing_charge_annual'].values[0]
current_ice_fuel_cost = summary_results_df.loc[current_cond]['vehicle_fuel_cost'].values[0]
current_total_energy_cost = (current_elec_cost+current_gas_cost+current_ice_fuel_cost)

current_elec_effective_cost_per_kWh = (summary_results_df.loc[current_cond]['grid_elec_import_cost'].values[0] - 
									  summary_results_df.loc[current_cond]['electricity_export_income'].values[0])/(summary_results_df.loc[current_cond]['grid_elec_import_Wh'].values[0]*0.001)


current_baseload_demand_Wh = summary_results_df.loc[current_cond]['electricity_demand_baseload_Wh'].values[0]
current_ev_demand_Wh = summary_results_df.loc[current_cond]['ev_charging_demand_Wh'].values[0]
current_heatpump_demand_Wh = summary_results_df.loc[current_cond]['electricity_demand_heatpump_Wh'].values[0]
current_solar_pv_generation_Wh = summary_results_df.loc[current_cond]['solar_pv_generation_Wh'].values[0]
current_grid_elec_export_Wh = summary_results_df.loc[current_cond]['grid_elec_export_Wh'].values[0]
current_solar_pv_self_consumed_Wh = current_solar_pv_generation_Wh - current_grid_elec_export_Wh

current_grid_elec_import_cost = summary_results_df.loc[current_cond]['grid_elec_import_cost'].values[0]
current_baseload_elec_cost = current_grid_elec_import_cost * (current_baseload_demand_Wh/(current_baseload_demand_Wh+current_ev_demand_Wh+current_heatpump_demand_Wh))
current_ev_elec_cost = current_grid_elec_import_cost * (current_ev_demand_Wh/(current_baseload_demand_Wh+current_ev_demand_Wh+current_heatpump_demand_Wh))
current_heatpump_elec_cost = current_grid_elec_import_cost * (current_heatpump_demand_Wh/(current_baseload_demand_Wh+current_ev_demand_Wh+current_heatpump_demand_Wh))
current_solar_pv_export_income = summary_results_df.loc[current_cond]['electricity_export_income'].values[0]
current_gas_standing_charge = summary_results_df.loc[current_cond]['gas_standing_charge_annual'].values[0]
current_gas_usage_cost = summary_results_df.loc[current_cond]['gas_import_cost'].values[0]


summary_results_df['Annual Savings'] = (current_total_energy_cost-
												summary_results_df['Annual Cost']).round(2)

summary_results_df['heating_upgrade_cost'] = 0.
heating_change_cond = (summary_results_df['heating_system_id']!=summary_results_df.loc[current_cond]['heating_system_id'].values[0])
summary_results_df['heating_upgrade_cost'].loc[heating_change_cond] = summary_results_df.loc[heating_change_cond]['heating_system_cost']

summary_results_df['solar_pv_upgrade_cost'] = 0.
solar_pv_change_cond = (summary_results_df['solar_pv_id']!=summary_results_df.loc[current_cond]['solar_pv_id'].values[0])
summary_results_df['solar_pv_upgrade_cost'].loc[solar_pv_change_cond] = summary_results_df.loc[solar_pv_change_cond]['solar_pv_cost']

summary_results_df['battery_storage_upgrade_cost'] = 0.
battery_storage_change_cond = (summary_results_df['battery_storage_id']!=summary_results_df.loc[current_cond]['battery_storage_id'].values[0])
summary_results_df['battery_storage_upgrade_cost'].loc[battery_storage_change_cond] = summary_results_df.loc[battery_storage_change_cond]['battery_storage_cost']

summary_results_df['ev_charger_upgrade_cost'] = 0.
ev_charger_change_cond = (summary_results_df['ev_charger_id']!=summary_results_df.loc[current_cond]['ev_charger_id'].values[0])
summary_results_df['ev_charger_upgrade_cost'].loc[ev_charger_change_cond] = summary_results_df.loc[ev_charger_change_cond]['ev_charger_cost']

summary_results_df['Investment'] = (summary_results_df['heating_upgrade_cost']+
											summary_results_df['solar_pv_upgrade_cost']+
											summary_results_df['battery_storage_upgrade_cost']+
											summary_results_df['ev_charger_upgrade_cost'])



future_potential_cond = ((summary_results_df['daily_miles_driven']==daily_miles_driven) &
		(summary_results_df['annual_electricity_consumption_kWh']==elec_kWh_slider) &
		(summary_results_df['annual_gas_consumption_kWh']==gas_kWh_slider) &
		(summary_results_df['heating_system_name'].isin(future_heating_system)) &
		(summary_results_df['vehicle_name'].isin(future_vehicle))&
		(summary_results_df['battery_storage_name'].isin(battery_storage_option))&
		(summary_results_df['solar_pv_name'].isin(solar_pv_option))&
		(summary_results_df['ev_charger_name'].isin(ev_charger_option))&
		(summary_results_df['tariff_name'].isin(energy_tariff_option))&		
		(summary_results_df['tariff_requires_smart_meter'].isin(smart_meter_option))&				
		(summary_results_df['solar_pv_system_size_Wp']<=solar_pv_size*1000.)&
		(summary_results_df['Investment']<=budget)
		)



n_scenarios = len(summary_results_df.loc[future_potential_cond].index)

col1, col2 = st.columns([1,2])
with col2:
 
	st.subheader('Select a scenario:',)	
	display_cols = ['Annual Savings','Investment','Products','scenario_id']
	gb = GridOptionsBuilder.from_dataframe(summary_results_df.loc[future_potential_cond][display_cols])	
	gb.configure_selection('single', pre_selected_rows=[0], use_checkbox=True)	
# 	gb.configure_selection('single', use_checkbox=True)		

	grid_response = AgGrid(
		summary_results_df.loc[future_potential_cond][display_cols],
		editable=False,
		gridOptions=gb.build(),
		data_return_mode="filtered_and_sorted",
		update_mode="selection_changed",	
# 		fit_columns_on_grid_load=True,
		fit_columns_on_grid_load=False,
		height=300,
		theme='material',
		wrapText=False,
# 		checkboxSelection=True,
		autoHeight=True
	)
	df = grid_response['data']
	print ('grid_response')	
	print (grid_response)

	selected = grid_response['selected_rows']
# 	If this is the first pass, and the user hasn't made a selection yet, we'll default 
# 	to the top row in the chart
	if len(selected) == 0:
		selected_scenario_id = df['scenario_id'].values[0]
# 	If the user has clicked on a different row, we should use the scenario_id from that row
	else:
		selected_df = pd.DataFrame(selected).apply(pd.to_numeric, errors='coerce')	
		selected_scenario_id = selected[0]['scenario_id']
		
	selected_future_scenario_cond = (summary_results_df['scenario_id']==selected_scenario_id)


future_elec_standing_charge = summary_results_df.loc[selected_future_scenario_cond]['electricity_standing_charge_annual'].values[0]

future_elec_cost = (summary_results_df.loc[selected_future_scenario_cond]['grid_elec_import_cost'].values[0] 
					- summary_results_df.loc[selected_future_scenario_cond]['electricity_export_income'].values[0]
					+ summary_results_df.loc[selected_future_scenario_cond]['electricity_standing_charge_annual'].values[0])
					
future_gas_cost = (summary_results_df.loc[selected_future_scenario_cond]['gas_import_cost'].values[0]
				   + summary_results_df.loc[selected_future_scenario_cond]['gas_standing_charge_annual'].values[0])
				   
future_ice_fuel_cost = summary_results_df.loc[selected_future_scenario_cond]['vehicle_fuel_cost'].values[0]

future_total_energy_cost = (future_elec_cost+future_gas_cost+future_ice_fuel_cost)



future_elec_effective_cost_per_kWh = (summary_results_df.loc[selected_future_scenario_cond]['grid_elec_import_cost'].values[0] - 
									  summary_results_df.loc[selected_future_scenario_cond]['electricity_export_income'].values[0])/(summary_results_df.loc[selected_future_scenario_cond]['grid_elec_import_Wh'].values[0]*0.001)


future_baseload_demand_Wh = summary_results_df.loc[selected_future_scenario_cond]['electricity_demand_baseload_Wh'].values[0]
future_ev_demand_Wh = summary_results_df.loc[selected_future_scenario_cond]['ev_charging_demand_Wh'].values[0]
future_heatpump_demand_Wh = summary_results_df.loc[selected_future_scenario_cond]['electricity_demand_heatpump_Wh'].values[0]
future_solar_pv_generation_Wh = summary_results_df.loc[selected_future_scenario_cond]['solar_pv_generation_Wh'].values[0]
future_grid_elec_export_Wh = summary_results_df.loc[selected_future_scenario_cond]['grid_elec_export_Wh'].values[0]
future_solar_pv_self_consumed_Wh = future_solar_pv_generation_Wh - future_grid_elec_export_Wh

future_grid_elec_import_cost = summary_results_df.loc[selected_future_scenario_cond]['grid_elec_import_cost'].values[0]
future_baseload_elec_cost = future_grid_elec_import_cost * (future_baseload_demand_Wh/(future_baseload_demand_Wh+future_ev_demand_Wh+future_heatpump_demand_Wh))

future_ev_elec_cost = future_grid_elec_import_cost * (future_ev_demand_Wh/(future_baseload_demand_Wh+future_ev_demand_Wh+future_heatpump_demand_Wh))
future_heatpump_elec_cost = future_grid_elec_import_cost * (future_heatpump_demand_Wh/(future_baseload_demand_Wh+future_ev_demand_Wh+future_heatpump_demand_Wh))
future_solar_pv_export_income = summary_results_df.loc[selected_future_scenario_cond]['electricity_export_income'].values[0]

future_gas_standing_charge = summary_results_df.loc[selected_future_scenario_cond]['gas_standing_charge_annual'].values[0]
future_gas_usage_cost = summary_results_df.loc[selected_future_scenario_cond]['gas_import_cost'].values[0]




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

with col1:
	if cost_savings > 0:
		st.subheader('Your Annual Energy Bill Would be £'+"{:.0f}".format(cost_savings)+' Lower')
	elif cost_savings == 0:
		st.subheader('Your Annual Energy Bill Would Be Unchanged')
	else:
		st.subheader('Your Annual Energy Bill Would be £'+"{:.0f}".format(-cost_savings)+' Higher')
	st_echarts(options=options)
	
st.markdown("""---""")

st.subheader('Products Installed')

col3, col4 = st.columns(2)

# TODO - Update this to a Ag-Grid table, for a cleaner feel and better alignment between rows!

with col3:
	st.write('Changes')
with col4:		
	st.write('Investment (£)')
if summary_results_df.loc[current_cond]['heating_system_name'].values[0] != summary_results_df.loc[selected_future_scenario_cond]['heating_system_name'].values[0]:
	with col3:
		st.write('')
		st.write('')		
		st.write('Upgrade',summary_results_df.loc[current_cond]['heating_system_name'].values[0],'to',summary_results_df.loc[selected_future_scenario_cond]['heating_system_name'].values[0])
	with col4:

		heating_system_installed_cost = st.number_input(label='', min_value=0, max_value=None, value=int(summary_results_df.loc[selected_future_scenario_cond]['heating_system_cost'].values[0]))
else:
	heating_system_installed_cost = 0

if summary_results_df.loc[current_cond]['solar_pv_name'].values[0] != summary_results_df.loc[selected_future_scenario_cond]['solar_pv_name'].values[0]:
	with col3:
		st.write('')
		st.write('')		
		st.write('Upgrade',summary_results_df.loc[current_cond]['solar_pv_name'].values[0],'to',summary_results_df.loc[selected_future_scenario_cond]['solar_pv_name'].values[0])
	with col4:
		solar_pv_installed_cost = st.number_input(label='', min_value=0, max_value=None, value=int(summary_results_df.loc[selected_future_scenario_cond]['solar_pv_cost'].values[0]))

else:
	solar_pv_installed_cost = 0

if summary_results_df.loc[current_cond]['battery_storage_name'].values[0] != summary_results_df.loc[selected_future_scenario_cond]['battery_storage_name'].values[0]:
	with col3:
# 		st.write('')
# 		st.write('')
		st.write('')
		st.write('')		
		st.write('Upgrade',summary_results_df.loc[current_cond]['battery_storage_name'].values[0],'to',summary_results_df.loc[selected_future_scenario_cond]['battery_storage_name'].values[0])
	with col4:
		battery_storage_installed_cost = st.number_input(label='', min_value=0, max_value=None, value=int(summary_results_df.loc[selected_future_scenario_cond]['battery_storage_cost'].values[0]))
else:
	battery_storage_installed_cost = 0


if summary_results_df.loc[current_cond]['ev_charger_name'].values[0] != summary_results_df.loc[selected_future_scenario_cond]['ev_charger_name'].values[0]:
	with col3:
		st.write('')
		st.write('')
		st.write('Upgrade',summary_results_df.loc[current_cond]['ev_charger_name'].values[0],'to',summary_results_df.loc[selected_future_scenario_cond]['ev_charger_name'].values[0])
	with col4:
		ev_charger_installed_cost = st.number_input(label='', min_value=0, max_value=None, value=int(summary_results_df.loc[selected_future_scenario_cond]['ev_charger_cost'].values[0]))	
else:
	ev_charger_installed_cost = 0


if summary_results_df.loc[current_cond]['vehicle_name'].values[0] != summary_results_df.loc[selected_future_scenario_cond]['vehicle_name'].values[0]:
	with col3:
		st.write('')
		st.write('')		
		st.write('Upgrade',summary_results_df.loc[current_cond]['vehicle_name'].values[0],'to',summary_results_df.loc[selected_future_scenario_cond]['vehicle_name'].values[0])
	with col4:
		st.write('')
		st.write('N/A - Assumes both vehicles are leased')




# 		st.write('Keep '+summary_results_df.loc[current_cond]['heating_system_name'].values[0])
# 	else:
# 		heating_system_installed_cost = st.number_input(label='Heating', min_value=0, max_value=None, value=int(summary_results_df.loc[selected_future_scenario_cond]['heating_system_cost'].values[0]))
# 		st.write('Upgrade',summary_results_df.loc[current_cond]['heating_system_name'].values[0],'to',summary_results_df.loc[selected_future_scenario_cond]['heating_system_name'].values[0])

# col3, col4, col5, col6, col7 = st.columns(5)
# with col3:
# 	
# 	if summary_results_df.loc[current_cond]['heating_system_name'].values[0] == summary_results_df.loc[selected_future_scenario_cond]['heating_system_name'].values[0]:
# 		heating_system_installed_cost = st.number_input(label='Heating', min_value=0, max_value=None, value=0)
# 		st.write('Keep '+summary_results_df.loc[current_cond]['heating_system_name'].values[0])
# 	else:
# 		heating_system_installed_cost = st.number_input(label='Heating', min_value=0, max_value=None, value=int(summary_results_df.loc[selected_future_scenario_cond]['heating_system_cost'].values[0]))
# 		st.write('Upgrade',summary_results_df.loc[current_cond]['heating_system_name'].values[0],'to',summary_results_df.loc[selected_future_scenario_cond]['heating_system_name'].values[0])
# 	
# with col4:
# 	if summary_results_df.loc[current_cond]['solar_pv_name'].values[0] == summary_results_df.loc[selected_future_scenario_cond]['solar_pv_name'].values[0]:
# 		solar_pv_installed_cost = st.number_input(label='Solar PV', min_value=0, max_value=None, value=0)
# 		st.write('Keep '+summary_results_df.loc[current_cond]['solar_pv_name'].values[0])
# 	else:
# 		solar_pv_installed_cost = st.number_input(label='Solar PV', min_value=0, max_value=None, value=int(summary_results_df.loc[selected_future_scenario_cond]['solar_pv_cost'].values[0]))
# 
# 		st.write('Upgrade',summary_results_df.loc[current_cond]['solar_pv_name'].values[0],'to',summary_results_df.loc[selected_future_scenario_cond]['solar_pv_name'].values[0])
# 
# with col5:
# 	if summary_results_df.loc[current_cond]['battery_storage_name'].values[0] == summary_results_df.loc[selected_future_scenario_cond]['battery_storage_name'].values[0]:
# 		battery_storage_installed_cost = st.number_input(label='Battery Storage', min_value=0, max_value=None, value=0)
# 		st.write('Keep '+summary_results_df.loc[current_cond]['battery_storage_name'].values[0])
# 	else:
# 		battery_storage_installed_cost = st.number_input(label='Battery Storage', min_value=0, max_value=None, value=int(summary_results_df.loc[selected_future_scenario_cond]['battery_storage_cost'].values[0]))
# 		st.write('Upgrade',summary_results_df.loc[current_cond]['battery_storage_name'].values[0],'to',summary_results_df.loc[selected_future_scenario_cond]['battery_storage_name'].values[0])
# 
# with col6:
# 
# 	if summary_results_df.loc[current_cond]['ev_charger_name'].values[0] == summary_results_df.loc[selected_future_scenario_cond]['ev_charger_name'].values[0]:
# 		ev_charger_installed_cost = st.number_input(label='EV Charger', min_value=0, max_value=None, value=0)
# 		st.write('Keep '+summary_results_df.loc[current_cond]['ev_charger_name'].values[0])
# 	else:
# 		ev_charger_installed_cost = st.number_input(label='EV Charger', min_value=0, max_value=None, value=int(summary_results_df.loc[selected_future_scenario_cond]['ev_charger_cost'].values[0]))	
# 		st.write('Upgrade',summary_results_df.loc[current_cond]['ev_charger_name'].values[0],'to',summary_results_df.loc[selected_future_scenario_cond]['ev_charger_name'].values[0])
# 
# 
# with col7:
# 
# 	energy_tariff_cost = st.number_input(label='Energy Tariff', min_value=0, max_value=None, value=0)
# 
# 	if summary_results_df.loc[current_cond]['tariff_name'].values[0] == summary_results_df.loc[selected_future_scenario_cond]['tariff_name'].values[0]:
# 		st.write('Keep '+summary_results_df.loc[current_cond]['tariff_name'].values[0])
# 	else:
# 
# 		st.write('Upgrade',summary_results_df.loc[current_cond]['tariff_name'].values[0],'to',summary_results_df.loc[selected_future_scenario_cond]['tariff_name'].values[0])
# 

total_investment_cost = (heating_system_installed_cost+
						 solar_pv_installed_cost+
						 battery_storage_installed_cost+
						 ev_charger_installed_cost)

payback_years = int(np.ceil(total_investment_cost/cost_savings))

st.write("---") # horizontal separator line.

col8, col9 = st.columns(2)

with col9:
# 	st.write('Total Investment',total_investment_cost)
	st.metric(label="Total Investment", value='£'+str(total_investment_cost))
with col8:
	if cost_savings <= 0.:
# 		payback_years = float('inf')
		st.metric(label="Payback", value='N/A')
	else:
		st.metric(label="Payback", value=str(payback_years)+' years')	
# 	total_cost = st.number_input(label='Total Investment', min_value=None, max_value=None, value=int(total_investment_cost), disabled=True)

# with col9:
# 	if cost_savings == 0.:
# 		payback_years = 0
# 	else:	
# 		payback_years = int(np.ceil(total_investment_cost/cost_savings))
# 	st.write('Investment Paid Back in',payback_years, 'Years')
# 	total_cost = st.number_input(label='Payback In Years', min_value=None, max_value=None, value=payback_years, disabled=True)
	
st.markdown("""---""")

st.subheader('Energy & Fuel Bill Breakdown')


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
elec_solar_pv_export = [-round(current_solar_pv_export_income,0), -round(future_solar_pv_export_income,0)]
elec_solar_pv_export = [x if x!=0 else '-' for x in elec_solar_pv_export]

options_en_bill = {
  "tooltip": {
    "trigger": 'axis',
    "axisPointer": {
      "type": 'shadow'
    }
  },
  "legend": {
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
      "name": 'Electricity: Exported Solar PV',
      "type": 'bar',
      "stack": 'Cost',      
      "label": {
        "show": True,
        "position": 'inside'
      },
      "emphasis": {
        "focus": 'series'
      },
      "data": elec_solar_pv_export
    }    
  ]
}
st_echarts(options=options_en_bill)

# st.write('Current Cost per kWh electricity',current_elec_effective_cost_per_kWh)
# st.write('Future Cost per kWh electricity',future_elec_effective_cost_per_kWh)


st.subheader('Electricity Demand')



options = {
  "tooltip": {
    "trigger": 'axis',
    "axisPointer": {
      "type": 'shadow'
    }
  },
  "legend": {
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
      "data": ['Current Electricity Demand (kWh)', 'Future Electricity Demand (kWh)']
    }
  ],
  "series": [
    {
      "name": 'Baseload',
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



st_echarts(options=options)



generation_export_options = {
  "tooltip": {
    "trigger": 'axis',
    "axisPointer": {
      "type": 'shadow'
    }
  },
  "legend": {
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
      "data": ['Current', 'Future']
    }
  ],
  "series": [
    {
      "name": 'Solar PV Generation - Self-Consumed',
      "type": 'bar',
      "stack": 'Total',      
      "label": {
        "show": True,
        "position": 'inside'
      },
      "emphasis": {
        "focus": 'series'
      },
      "data": [round(current_solar_pv_self_consumed_Wh/1000.,0), round(future_solar_pv_self_consumed_Wh/1000.,0)]
    },   
    {
      "name": 'Solar PV Generation - Exported',
      "type": 'bar',
      "stack": 'Total',
      "label": {
        "show": True
      },
      "emphasis": {
        "focus": 'series'
      },
      "data": [round(current_grid_elec_export_Wh/1000.,0), round(future_grid_elec_export_Wh/1000.,0)]
    }
  ]
}




if (current_solar_pv_system != 'No Solar PV') | (future_solar_pv_system != 'No'):
	st.subheader('Generation & Export')
	st_echarts(options=generation_export_options)
	
# else:
# 	st.markdown("<h1 style='text-align: center; color: red;'>Some title</h1>", unsafe_allow_html=True)

# 	st.write('No Solar PV')

