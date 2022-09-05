import streamlit as st
import numpy as np
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder
from streamlit_echarts import st_echarts


st.set_page_config(layout="wide")

summary_results_df = pd.read_csv('summary_results.csv')

typical_demand_profile_df = pd.read_csv('typical_demand_profile_results.csv')

cols = ['heating_system_name','solar_pv_name','battery_storage_name','vehicle_name','ev_charger_name','tariff_name']
summary_results_df['Products'] = summary_results_df[cols].apply(lambda row: '\n'.join(row.values.astype(str)), axis=1)

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


with st.sidebar.expander("Heating"):
	current_heating_system = st.selectbox(
		 'Current',
		 summary_results_df['heating_system_name'].unique()
		 )
	st.markdown("""---""")

	future_heating_system = st.multiselect(
				'Future',
				summary_results_df['heating_system_name'].unique(),
				summary_results_df['heating_system_name'].unique()
				)


with st.sidebar.expander("Battery Storage"):

	current_battery_storage_system = st.selectbox(
		 'Current',
		 summary_results_df['battery_storage_name'].unique()
		 )
		 
	st.markdown("""---""")

	battery_storage_option = st.multiselect(
				'Future',
				summary_results_df['battery_storage_name'].unique(),
				summary_results_df['battery_storage_name'].unique()
				)

with st.sidebar.expander("Solar PV"):
	current_solar_pv_system = st.selectbox(
		 'Current',
		 summary_results_df['solar_pv_name'].unique()
		 )
	st.markdown("""---""")

	solar_pv_option = st.multiselect(
				'Future',
				summary_results_df['solar_pv_name'].unique(),
				summary_results_df['solar_pv_name'].unique()
				)

with st.sidebar.expander("Vehicle"):
	current_vehicle = st.selectbox(
		 'Current',
		 summary_results_df['vehicle_name'].unique()
		 )
	st.markdown("""---""")

	future_vehicle = st.multiselect(
				'Future',
				summary_results_df['vehicle_name'].unique(),
				summary_results_df['vehicle_name'].unique()
				)

with st.sidebar.expander("EV Charger"):
	current_ev_charger = st.selectbox(
		 'Current',
		 summary_results_df['ev_charger_name'].unique()
		 )

	st.markdown("""---""")

	ev_charger_option = st.multiselect(
				'Future',
				summary_results_df['ev_charger_name'].unique(),
				summary_results_df['ev_charger_name'].unique()
				)

with st.sidebar.expander("Energy Tariff"):
	current_energy_tariff = st.selectbox(
		 'Current',
		 summary_results_df['tariff_name'].unique()
		 )

	st.markdown("""---""")
	energy_tariff_option = st.multiselect(
				'Future',
				summary_results_df['tariff_name'].unique(),
				summary_results_df['tariff_name'].unique()
				)

with st.expander("ℹ️ - Getting Started", expanded=True):

    st.write(
        """     
-   The *DEBRA* app is an easy-to-use interface built in Streamlit for UK households to find the most profitable low-carbon upgrades for their properties
-   Use the left-side navigation menu to select your current energy usage; installed products; which products you would consider upgrading to; and the budget you have in mind
-   Use the table below to pick a scenario you're interested in, and see how those savings are achieved.
-   Support & contribute to this [open-source project on GitHub](https://github.com/cutmyenergybill/domestic-energy-bill-reduction-app/) - new products, ideas and thoughts welcome!
	    """
    )

    st.markdown("") 

user_selection_error = False
if (len(future_heating_system) == 0):
	st.write("""
-	Please select at least 1 option for your future Heating System
""")
	user_selection_error = True

if (len(battery_storage_option) == 0):
	st.write("""
-	Please select at least 1 option for your future Battery Storage
""")
	user_selection_error = True

if (len(solar_pv_option) == 0):
	st.write("""
-	Please select at least 1 option for your future Solar PV
""")
	user_selection_error = True

if (len(future_vehicle) == 0):
	st.write("""
-	Please select at least 1 option for your future Vehicle
""")
	user_selection_error = True

if (len(ev_charger_option) == 0):
	st.write("""
-	Please select at least 1 option for your future EV Charger
""")
	user_selection_error = True

if (len(energy_tariff_option) == 0):
	st.write("""
-	Please select at least 1 option for your future Energy Tariff
""")
	user_selection_error = True



budget = st.sidebar.slider('Budget (£)', 0, 50000, 30000, step=500)
    
vehicle_fuel_cost_per_litre = st.sidebar.slider('Vehicle Fuel Cost (£/litre)', 1.00, 2.50, 1.80, step=0.01)

st.sidebar.write('Kindly supported by [Climate Subak](https://climatesubak.org/)')

st.sidebar.write('Contribute to the [GitHub Project](https://github.com/cutmyenergybill/domestic-energy-bill-reduction-app/)')


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


summary_results_df['Heating'] = ''
summary_results_df['Heating'].loc[summary_results_df['heating_system_name']!=current_heating_system] = '✓'

summary_results_df['Battery'] = ''
summary_results_df['Battery'].loc[summary_results_df['battery_storage_name']!=current_battery_storage_system] = '✓'

summary_results_df['Solar PV'] = ''
summary_results_df['Solar PV'].loc[summary_results_df['solar_pv_name']!=current_solar_pv_system] = '✓'

summary_results_df['Tariff'] = ''
summary_results_df['Tariff'].loc[summary_results_df['tariff_name']!=current_energy_tariff] = '✓'

summary_results_df['EV Charger'] = ''
summary_results_df['EV Charger'].loc[summary_results_df['ev_charger_name']!=current_ev_charger] = '✓'

summary_results_df['Vehicle'] = ''
summary_results_df['Vehicle'].loc[summary_results_df['vehicle_name']!=current_vehicle] = '✓'



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
												summary_results_df['Annual Cost']).round(0)

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
# 		(summary_results_df['tariff_requires_smart_meter'].isin(smart_meter_option))&				
# 		(summary_results_df['solar_pv_system_size_Wp']<=solar_pv_size*1000.)&
		(summary_results_df['Investment']<=budget)
		)



n_scenarios = len(summary_results_df.loc[future_potential_cond].index)
 

 
st.subheader('Select Your Upgrades:',)	
display_cols = ['Annual Savings','Investment','Products','scenario_id']
display_cols = ['Annual Savings','Heating','Battery','Solar PV','Tariff','Vehicle','EV Charger','scenario_id']	
gb = GridOptionsBuilder.from_dataframe(summary_results_df.loc[future_potential_cond][display_cols])	
gb.configure_selection('single', pre_selected_rows=[0], use_checkbox=True)	
# 	gb.configure_selection('single', use_checkbox=True)		

grid_response = AgGrid(
	summary_results_df.loc[future_potential_cond][display_cols],
	editable=False,
	gridOptions=gb.build(),
	data_return_mode="filtered_and_sorted",
	update_mode="selection_changed",	
# 	fit_columns_on_grid_load=True,
	fit_columns_on_grid_load=False,
	height=200,
	theme='streamlit',
# 	wrapText=True,
	wrapText=False,		
# 		checkboxSelection=True,
# 		autoHeight=True
)
df = grid_response['data']
# print ('grid_response')	
# print (grid_response)





selected = grid_response['selected_rows']
# 	If this is the first pass, and the user hasn't made a selection yet, we'll default 
# 	to the top row in the chart

# 	If there isn't a scenario that fits the user's preferences, then something else is wrong!

if user_selection_error == False:

	if len(selected) == 0:
		selected_scenario_id = df['scenario_id'].values[0]
	# 	If the user has clicked on a different row, we should use the scenario_id from that row
	else:
		selected_df = pd.DataFrame(selected).apply(pd.to_numeric, errors='coerce')	
		selected_scenario_id = selected[0]['scenario_id']
	
	selected_future_scenario_cond = (summary_results_df['scenario_id']==selected_scenario_id)


	future_solar_pv_system = summary_results_df['solar_pv_name'].loc[selected_future_scenario_cond].values[0]

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
		product_to.append(summary_results_df.loc[selected_future_scenario_cond]['solar_pv_name'].values[0])
		product_investment_cost.append(int(summary_results_df.loc[selected_future_scenario_cond]['solar_pv_cost'].values[0]))

	if summary_results_df.loc[current_cond]['battery_storage_name'].values[0] != summary_results_df.loc[selected_future_scenario_cond]['battery_storage_name'].values[0]:
		product_type.append('Battery Storage')
		product_from.append(summary_results_df.loc[current_cond]['battery_storage_name'].values[0])
		product_to.append(summary_results_df.loc[selected_future_scenario_cond]['battery_storage_name'].values[0])
		product_investment_cost.append(int(summary_results_df.loc[selected_future_scenario_cond]['battery_storage_cost'].values[0]))
				
	if summary_results_df.loc[current_cond]['vehicle_name'].values[0] != summary_results_df.loc[selected_future_scenario_cond]['vehicle_name'].values[0]:
		product_type.append('Vehicle')
		product_from.append(summary_results_df.loc[current_cond]['vehicle_name'].values[0])
		product_to.append(summary_results_df.loc[selected_future_scenario_cond]['vehicle_name'].values[0])
		product_investment_cost.append(0)		

	if summary_results_df.loc[current_cond]['ev_charger_name'].values[0] != summary_results_df.loc[selected_future_scenario_cond]['ev_charger_name'].values[0]:

		product_type.append('EV Charging')
		product_from.append(summary_results_df.loc[current_cond]['ev_charger_name'].values[0])
		product_to.append(summary_results_df.loc[selected_future_scenario_cond]['ev_charger_name'].values[0])
		product_investment_cost.append(int(summary_results_df.loc[selected_future_scenario_cond]['ev_charger_cost'].values[0]))
		
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


	st.markdown("""---""")						   
						   
	col3, col4 = st.columns([3,1])						   

	with col3:
		if cost_savings > 0:
			st.subheader('Your Annual Energy Bill Would be £'+"{:.0f}".format(cost_savings)+' Lower')
		elif cost_savings == 0:
			st.subheader('Your Annual Energy Bill Would Be Unchanged')
		else:
			st.subheader('Your Annual Energy Bill Would be £'+"{:.0f}".format(-cost_savings)+' Higher')




		changes_df = pd.DataFrame({'Upgrade': product_type, 
								   'From': product_from,
								   'To': product_to,
								   'Cost': product_investment_cost
								   })
		st_echarts(options=options)

	with col4:


		change_gas_cost = future_gas_cost-current_gas_cost
		change_elec_cost = future_elec_cost-current_elec_cost
		change_ice_fuel_cost = future_ice_fuel_cost-current_ice_fuel_cost	

		st.subheader('Natural Gas:')
	
		if change_gas_cost < 0:
			st.write('Saving £'+"{:.2f}".format(abs(change_gas_cost))+'/year')		

		if change_gas_cost > 0:
			st.write('Adding £'+"{:.2f}".format(abs(change_gas_cost))+'/year')

		if change_gas_cost == 0:	
			st.write('No Change')

		st.subheader('Electricity')
		if change_elec_cost < 0:
			st.write('Saving £'+"{:.2f}".format(abs(change_elec_cost))+'/year')		

		if change_elec_cost > 0:
			st.write('Adding £'+"{:.2f}".format(abs(change_elec_cost))+'/year')

		if change_elec_cost == 0:	
			st.write('No Change')
		
		st.subheader('Petrol/Diesel')
		if change_ice_fuel_cost < 0:
			st.write('Saving £'+"{:.2f}".format(abs(change_ice_fuel_cost))+'/year')		

		if change_ice_fuel_cost > 0:
			st.write('Adding £'+"{:.2f}".format(abs(change_ice_fuel_cost))+'/year')
		
		if change_ice_fuel_cost == 0:	
			st.write('No Change')
				

	st.subheader('Product Upgrades')

	st.markdown("_Adjust investment cost manually by editing the numbers in the cell - payback and total investment figures will update_", unsafe_allow_html=False)


	grid_return = AgGrid(changes_df, editable=True,
						fit_columns_on_grid_load=True,
						width=[5,1,1,1],
						height=100+(45*len(changes_df.index)),
						theme='material',
						wrapText=False,
				# 		checkboxSelection=True,
						autoHeight=True
						)

	changes_df = grid_return['data']

	total_investment_cost = (changes_df['Cost'].astype(int).sum())

	if cost_savings > 0:
		payback_years = int(np.ceil(total_investment_cost/cost_savings))
	else:
		payback_years = 0


	col8, col9, col10 = st.columns(3)

	with col8:
		st.metric(label="Annual Savings", value='£'+"{:.2f}".format(cost_savings))

	with col9:
		st.metric(label="Total Investment", value='£'+str(total_investment_cost))

	with col10:
		if cost_savings <= 0.:
			st.metric(label="Payback", value='N/A')
		else:
			st.metric(label="Payback", value=str(payback_years)+' years')	



	st.markdown("*_Assumes same cost to lease EV as a Petrol/Diesel car_", unsafe_allow_html=False)
	st.markdown("*_No cost associated with switching to new tariff, assumes smart meter can be installed free-of-charge_", unsafe_allow_html=False)


	
	st.markdown("""---""")

	st.subheader('Energy & Fuel Bill Breakdown')

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
	# with col10:
	# 	st.write("")
	# 
	# with col11:
	# 	st.write("""    
	# 	- Gas consumption down by XkWh
	# 	- Electricity consumption up by YkWh
	# 	- This is because of Heat Pump (+Z kWh) and EV (+ZZ kWh)
	# 	"""
	# 	)

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


	if (current_solar_pv_system != 'No Solar PV') & (future_solar_pv_system != 'No Solar PV'):
		st.subheader('Generation & Export')
		st_echarts(options=generation_export_options)
	
	st.subheader('Half-Hourly Profiles')

# 	print (typical_demand_profile_df.loc[typical_demand_profile_df['scenario_id']==selected_scenario_id]['time'].values)	
# 	print (typical_demand_profile_df.loc[typical_demand_profile_df['scenario_id']==selected_scenario_id]['grid_elec_import_Wh'].values)
# 	print (typical_demand_profile_df.loc[typical_demand_profile_df['scenario_id']==selected_scenario_id]['solar_pv_generation_Wh'].values)

	option = {
		"title": {"text": "Typical Day Electricity Demand (kWh)"},
		"tooltip": {"trigger": "axis"},
		"legend": {"data": ["Baseload","Heat Pump","EV Charging"],
# 					"align":"right", 
					"x":"right",
					"y":"top"
					},
		"grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
# 		"toolbox": {"feature": {"saveAsImage": {}}},	
		"xAxis": {
			"type": "category",
			"data": list(typical_demand_profile_df.loc[typical_demand_profile_df['scenario_id']==selected_scenario_id]['time'].values),
		},
		"yAxis": {"type": "value", 
# 					"axisLabel": {"rotate": "90"},
				},
		"series": [
			{"name":'Baseload',"data": list(typical_demand_profile_df.loc[typical_demand_profile_df['scenario_id']==selected_scenario_id]['electricity_demand_baseload_Wh'].values/1000.), "type": "bar","stack": "Total"},
			{"name":'Heat Pump',"data": list(typical_demand_profile_df.loc[typical_demand_profile_df['scenario_id']==selected_scenario_id]['electricity_demand_heatpump_Wh'].values/1000.), "type": "bar","stack": "Total"},						
			{"name":'EV Charging',"data": list(typical_demand_profile_df.loc[typical_demand_profile_df['scenario_id']==selected_scenario_id]['ev_charging_demand_Wh'].values/1000.), "type": "bar","stack": "Total"},						
# 			{"name":'solar_pv_generation_Wh',"data": list(typical_demand_profile_df.loc[typical_demand_profile_df['scenario_id']==selected_scenario_id]['solar_pv_generation_Wh'].values), "type": "line"}
			
			],
	}
	st_echarts(
		options=option, 
		height="300px",
	)



	option = {
		"title": {"text": "Typical Day Grid Import (kWh) & Electricity Price (£/kWh)"},
		"tooltip": {"trigger": "axis"},
		"legend": {"data": ["Grid Import","Electricity Price"],
# 					"align":"right",
					"x":"right",
					"y":"top"},
		"grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
# 		"toolbox": {"feature": {"saveAsImage": {}}},	
		"xAxis": {
			"type": "category",
			"data": list(typical_demand_profile_df.loc[typical_demand_profile_df['scenario_id']==selected_scenario_id]['time'].values),
		},
		"yAxis": [{"type": "value", "position": "left",

# 					"axisLabel": {"rotate": "90"},
					},
					{"type": "value", "position": "right",}
					],
		"series": [
			{"name":'Grid Import',"data": list(typical_demand_profile_df.loc[typical_demand_profile_df['scenario_id']==selected_scenario_id]['grid_elec_import_Wh'].values/1000.), "type": "bar", "yAxisIndex": 0},
			{"name":'Electricity Price',"data": list(typical_demand_profile_df.loc[typical_demand_profile_df['scenario_id']==selected_scenario_id]['electricity_unit_rate_per_kWh'].values), "type": "line", "yAxisIndex": 1},
# 			{"name":'Heat Pump',"data": list(typical_demand_profile_df.loc[typical_demand_profile_df['scenario_id']==selected_scenario_id]['electricity_demand_heatpump_Wh'].values/1000.), "type": "bar","stack": "Total"},						
# 			{"name":'EV Charging',"data": list(typical_demand_profile_df.loc[typical_demand_profile_df['scenario_id']==selected_scenario_id]['ev_charging_demand_Wh'].values/1000.), "type": "bar","stack": "Total"},			
			
# 			{"name":'solar_pv_generation_Wh',"data": list(typical_demand_profile_df.loc[typical_demand_profile_df['scenario_id']==selected_scenario_id]['solar_pv_generation_Wh'].values), "type": "line"}
			
			],	
	}	
	st_echarts(
		options=option, 
		height="300px",
	)
# 	grid_elec_import_Wh
# 	electricity_unit_rate_per_kWh
# 
# 	battery_generation_Wh
# 	battery_charging_demand_Wh


else:
	st.write('No analysis available - please check inputs and try again!')
	