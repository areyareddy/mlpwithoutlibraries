inputs = [0, 1, 3, 16, 4]
# inputs that the mystery function was given
true_vals = [2, 5, 11, 50, 14]
# the outputs that the mystery function returned for those inputs
# since we're trying to mimic the mystery function, we can call these "true values"

w = 4 
b = 3
# randomly chosen initial values for w and b, our weight and bias. 
# Currently the neural network consists of a nn.Linear(1, 1).
# In math form, f(x) = wx + b, where w = 4 and b = 3.

num_epochs = 1000
def L_w(w, b, input_num, true_val):
    return 2*(w*input_num + b - true_val)*input_num
def L_b(w, b, input_num, true_val):
    return 2*(w*input_num + b - true_val)
# These are partial derivatives for w and b, respectively. 
# They answer "if you change [w/b] by dx, what multiplier of that will the loss change?"
# So if your value of L_w is 0.5, then if you change w by 0.0001, the loss will change by approximately 0.00001*0.5.
# Here's how they're calculated: 

# The loss function (which here is mean squared error) is (guessed_val - true_val)^2. 
# And we know what guessed_val is: the output of our function, wx+b. 
# So we can rewrite the loss function as (wx + b - true_val)^2.
# Let's go back to the "changing w by 0.0001" example from earlier. 
# What it actually means to change w by 0.0001 is that we add 0.0001 to the value of w, and keep b constant. 
# Since b does not change when we change w, we can treat b as an actual constant, no different than a number like 5.
# So when we take the derivative of our loss function with respect to w, we can treat b... like an actual constant!

# If our function was 10w + bw^2 for example, and we wanted to take the derivative with respect to w:
# We treat b as if it's a new number that has been added to the list of numbers. 
# Derivative of 10w + 5w^2 --> 10+2*5*w, derivative of 10w + 3w^2 --> 10+2*3*w, 
# and likewise, derivative of 10w + bw^2 --> 10+2*b*w.

# Hopefully you understand partial derivatives better now, and why we can just treat other variables like constants (they are!)
# Our actual loss function has 4 variables, the input, the true value, the value of b, and the value of w.
# The only two we can change are w and b, so we take the derivative with respect to those two variables, treating the input as a constant.
# L(w, b, i, v) = (wi + b - v)^2 
# By chain rule, this is 
# [d/dw outer function](inner part) * d/dw (inner part) 
# outer function = [thing]^2, derivative = 2[thing]
# Remember to multiply by d/dw (inner part)! 
# L_w = 2(wi + b - v) * d/dw (wi + b - v) = 2*(wi + b - v)*i
# L_b = 2(wi + b - v) * d/db (wi + b - v) = 2*(wi + b - v)*1
# And that's exactly the equations we have here! 

for c in range(1, num_epochs+1):
    total_L_w = 0
    total_L_b = 0
    # We want to calculate the sum of all optimizations made to w and b for all inputs
    # Because we want to optimize w and b across all inputs, not just one. 
    for i, v in zip(inputs, true_vals):  
        total_L_w += L_w(w, b, i, v)
        total_L_b += L_b(w, b, i, v)
    print(f"Epoch: {c}, Total L_w: {total_L_w}, Total L_b: {total_L_b}")
    w -= 0.001 * total_L_w
    b -= 0.001 * total_L_b
    # The core idea of ML is that if we continuously optimize our parameters on the current values, we will optimize them.
    # The derivative of the function represents the slope of the tangent line, 
    # or which direction to go to increase the function the most. 
    # Since we have a LOSS function, we take the exact opposite of this, to DEcrease the function the most.
    # We also don't want to step fully in this direction because that overshoots a lot.
    # So we just step 0.001*value of the derivative.  
    print(f"Epoch: {c}, w={w}, b={b}")
