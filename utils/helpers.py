"""
Helper functions for the Real Estate App.
"""

def format_price(value):
    """
    Format price to include commas and currency symbol.

    Parameters:
        value (float): The predicted price

    Returns:
        str: Formatted price string
    """
    try:
        return f"${value:,.2f}"
    except Exception as e:
        raise Exception(f"Error formatting price: {e}")
