import Base.+

struct MyComplex
    Re
    Im
end

function my_times(a::MyComplex, b::MyComplex)
    real_part = a.Re * b.Re - a.Im * b.Im
    imaginary_part = a.Re * b.Im + a.Im * b.Re
    return MyComplex(real_part, imaginary_part)
end

# test-case
x = MyComplex(1.0, 2.0)
y = MyComplex(3.0, 4.0)
z = my_times(x,y)

# compare to Julia's built in complex arithmetic!
x_julia = 1.0 + 2.0*im  # defines a complex number: 1 + 1i
y_julia = 3.0 + 4.0*im  # defines a complex number: 0.5 + 3i
z_julia = x_julia * y_julia

# Implement function my_add that adds
# two objects of type MyComplex and returns the result as such
function my_add(a::MyComplex, b::MyComplex)
    return MyComplex(a.Re + b.Re, a.Im + b.Im)
end

my_add(x,y)
x_julia + y_julia
# create a method for Julia's + function to work on your type here
function +(a::MyComplex, b::MyComplex)
    return my_add(x,y)
end

# run the following lines to test your implementation
z = x + y
sum([x,y])