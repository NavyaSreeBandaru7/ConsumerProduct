%%writefile app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import spacy

# Load NLP model
nlp = spacy.load("en_core_web_sm")

def extract_equation_params(text):
    """
    Extracts demand and supply equation parameters from a natural language query.
    """
    doc = nlp(text.lower())  
    numbers = [float(token.text) for token in doc if token.like_num]  

    if len(numbers) < 4:
        return None  

    demand_intercept, demand_slope, supply_intercept, supply_slope = numbers[:4]
    demand_slope = -abs(demand_slope)  # Ensure demand slope is negative

    return demand_intercept, demand_slope, supply_intercept, supply_slope

def calculate_equilibrium(demand_intercept, demand_slope, supply_intercept, supply_slope):
    """
    Computes Competitive Equilibrium: Quantity and Price.
    Also calculates CS, PS, and SW for the competitive market.
    """
    quantity_eq = (demand_intercept - supply_intercept) / (supply_slope - demand_slope)
    price_eq = demand_intercept + demand_slope * quantity_eq

    # Compute competitive surpluses
    cs_eq = (demand_intercept - price_eq) * quantity_eq / 2
    ps_eq = (price_eq - supply_intercept) * quantity_eq / 2
    sw_eq = cs_eq + ps_eq

    plt.axhline(price_eq, linestyle="dashed", color="black", alpha=0.7, label=f"Original Equilibrium Price = {price_eq}")


        # Generate supply & demand curves
    q_range = np.linspace(0, quantity_eq * 1.5, 100)
    demand_curve = demand_intercept + demand_slope * q_range
    supply_curve = supply_intercept + supply_slope * q_range

   
    return round(quantity_eq, 2), round(price_eq, 2), round(cs_eq, 2), round(ps_eq, 2), round(sw_eq, 2)

def apply_price_ceiling(price_ceiling, demand_intercept, demand_slope, supply_intercept, supply_slope):
    """
    Computes:
    - New Quantity Supplied under the Price Ceiling (Q_s)
    - New Market Price (P_d)
    - Marginal Cost at Q_s (MC)
    - Correct Producer Surplus using (P_d - MC) * Q_s
    """
    quantity_supplied = (price_ceiling - supply_intercept) / supply_slope  # Q_s
    new_market_price = demand_intercept + demand_slope * quantity_supplied  # P_d
    marginal_cost = supply_intercept + supply_slope * quantity_supplied  # MC

    # Compute Surpluses
    consumer_surplus = (demand_intercept - new_market_price) * quantity_supplied / 2
    producer_surplus = (new_market_price - marginal_cost) * quantity_supplied + (0.5 * quantity_supplied * (price_ceiling - supply_intercept))  # (P_d - MC) * Q_s
    social_welfare = consumer_surplus + producer_surplus

    return round(quantity_supplied, 2), round(new_market_price, 2), round(marginal_cost, 2), round(consumer_surplus, 2), round(producer_surplus, 2), round(social_welfare, 2)
def plot_graph(demand_slope, demand_intercept, supply_slope, supply_intercept, quantity_eq, price_eq, price_ceiling, new_quantity, new_price, marginal_cost):
    q_range = np.linspace(0, max(quantity_eq, new_quantity) * 1.5, 100)
    demand_curve = demand_intercept + demand_slope * q_range
    supply_curve = supply_intercept + supply_slope * q_range

    plt.figure(figsize=(8, 6))
    plt.plot(q_range, demand_curve, label="Demand Curve", color="blue")
    plt.plot(q_range, supply_curve, label="Supply Curve", color="green")

    plt.scatter(quantity_eq, price_eq, color="red", zorder=5, label=f"Orig Comp Equil ({quantity_eq}, {price_eq})")

    if price_ceiling is not None:
        plt.axhline(price_ceiling, linestyle="dashed", color="orange", label=f"Price Ceiling = {price_ceiling}")
       
        if new_quantity is not None and new_price is not None:
            plt.scatter(new_quantity, new_price, color="purple", zorder=5, label=f"New Equil ({new_quantity}, {new_price})")
            plt.axvline(new_quantity, linestyle="dashed", color="purple", label=f"Quantity Supplied = {new_quantity}")

            # **Fill Producer Surplus (PS) correctly: Rectangle + Triangle**
            # 1. Fill the rectangle (MC to P_d)
            plt.fill_between([0, new_quantity], marginal_cost, new_price, color='purple', alpha=0.3)

            # 2. Fill the triangle (MC up to Price Ceiling) **Properly**
            # Debugging: Print the triangle coordinates in Streamlit
            #st.write("### Debugging Triangle Coordinates:")
            #st.write(f"Triangle X-coordinates: [0, 0, {new_quantity}]")
            #st.write(f"Triangle Y-coordinates: [{supply_intercept}, {price_ceiling}, {price_ceiling}]")

            triangle_x = [0, 0, new_quantity]  # (Left-bottom, Left-top, Right-top)
            triangle_y = [supply_intercept, price_ceiling, price_ceiling]  # (Bottom-left, Top-left, Top-right)
            plt.fill(triangle_x, triangle_y, color='purple', alpha=0.3, label="New PS")

            # New CS
            plt.fill([0, 0, new_quantity], [new_price, demand_intercept, new_price], color='green', alpha=0.3, label="New CS")
            #plt.text(new_quantity / 2, (marginal_cost + price_ceiling) / 2, "New PS", color='white', fontsize=15, ha='center', va='center')

            # CS Loss
            plt.fill([new_quantity, new_quantity, quantity_eq], [new_price, price_eq, price_eq], color='red', alpha=0.3, label="CS Loss")

            # PS Loss
            plt.fill([new_quantity, new_quantity, quantity_eq], [marginal_cost, price_eq, price_eq], color='orange', alpha=0.3, label="PS Loss")
 
            # plot original equilibrium again
            plt.axhline(price_eq, linestyle="dashed", color="black", alpha=0.7, label=f"Original Equilibrium Price = {price_eq}")

    plt.xlabel("Quantity")
    plt.ylabel("Price")
    plt.legend()
    plt.title("Consumer & Producer Surplus Visualization")
    st.pyplot(plt)

# Streamlit Interface
st.title("NRM-Bot: Price Ceiling and Market Equilibrium")

st.write("Enter a question in natural language (e.g., 'Find the equilibrium for a demand curve with intercept 100 and slope -0.5 and a supply curve with an intercept of 15 and a slope of 0.35.')")
user_query = st.text_area("Type your question here:", "")
price_ceiling = st.number_input("Enter a price ceiling (default: 1)", min_value=0.0, step=0.1, value=1.0)

if st.button("Solve"):
    params = extract_equation_params(user_query)

    if params:
        demand_intercept, demand_slope, supply_intercept, supply_slope = params
        st.info(f"**Extracted Values:**\n Demand Intercept: {demand_intercept}\n Demand Slope: {demand_slope}\n Supply Intercept: {supply_intercept}\n Supply Slope: {supply_slope}")

        # Compute Competitive Equilibrium
        quantity_eq, price_eq, cs_eq, ps_eq, sw_eq = calculate_equilibrium(demand_intercept, demand_slope, supply_intercept, supply_slope)

        # Compute Price Ceiling Equilibrium
        new_quantity, new_price, marginal_cost, new_cs, new_ps, new_sw = apply_price_ceiling(price_ceiling, demand_intercept, demand_slope, supply_intercept, supply_slope)

        # ✅ Print Competitive Equilibrium values
        st.success(f"**Competitive Equilibrium:**\n"
                   f"**Quantity:** {quantity_eq} units\n"
                   f"**Price:** ${price_eq}\n"
                   f"**Consumer Surplus (CS):** ${cs_eq}\n"
                   f"**Producer Surplus (PS):** ${ps_eq}\n"
                   f"**Total Social Welfare (SW):** ${sw_eq}")

        # ✅ Print Price Ceiling Equilibrium values
        st.success(f"**With Price Ceiling at {price_ceiling}:**\n"
                   f"**New Quantity Supplied:** {new_quantity} units\n"
                   f"**New Market Price Consumers Pay:** ${new_price}\n"
                   f"**Marginal Cost of Production:** ${marginal_cost}\n"
                   f"**New Consumer Surplus (CS):** ${new_cs}\n"
                   f"**New Producer Surplus (PS):** ${new_ps}\n"
                   f"**New Total Social Welfare (SW):** ${new_sw}")

        # Create comparison table with changes
        surplus_data = pd.DataFrame({
        "Scenario": ["Competitive Equilibrium", "With Price Ceiling", "Change"],
        "Consumer Surplus": [cs_eq, new_cs, new_cs - cs_eq],
        "Producer Surplus": [ps_eq, new_ps, new_ps - ps_eq],
        "Total Social Welfare": [sw_eq, new_sw, new_sw - sw_eq]
        })


        st.table(surplus_data)

        plot_graph(demand_slope, demand_intercept, supply_slope, supply_intercept, quantity_eq, price_eq, price_ceiling, new_quantity, new_price, marginal_cost)

    else:
        st.error("Could not extract demand and supply equation parameters. Please format your question correctly.")
