import matplotlib.pyplot as plt
import numpy as np

# Linear solver
def my_linfit(x, y):
    N = len(x)
    
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x2 = np.sum(x**2)
    sum_xy = np.sum(x * y)
    
    # Calculate 'a' directly using the formula
    a = (sum_xy * N - sum_x * sum_y) / (sum_x2 * N - sum_x**2)
    
    # Now calculate 'b' using the formula
    b = (sum_y - a * sum_x) / N
    return a, b

# Callback function to collect points on mouse click
def onclick(event, x_data, y_data, fig, ax):
    if event.button == 1:  # Left mouse button
        x_data.append(event.xdata)
        y_data.append(event.ydata)
        ax.plot(event.xdata, event.ydata, 'kx')
        fig.canvas.draw()
    elif event.button == 3:  # Right mouse button
        # Stop collecting points and plot the linear fit
        fig.canvas.mpl_disconnect(cid)
        
        # Convert lists to numpy arrays
        x = np.array(x_data)
        y = np.array(y_data)
        
        # Calculate linear fit
        a, b = my_linfit(x, y)
        
        # Plot the linear fit
        xp = np.linspace(min(x), max(x), 100)
        ax.plot(xp, a * xp + b, 'r-')
        print(f"My fit: a={a:.2f}, b={b:.2f}")
        plt.draw()

# Main
if __name__ == "__main__":
    x_data = []
    y_data = []
    
    fig, ax = plt.subplots()
    ax.set_title("Click to add points (right-click to stop)")
    cid = fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, x_data, y_data, fig, ax))
    
    plt.show()
