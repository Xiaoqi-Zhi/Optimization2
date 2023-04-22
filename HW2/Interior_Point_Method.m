syms x1 x2 t;
f = t*(-5*x1-x2)-log(x1+x1-5)-log(2*x1+0.5*x2-8)-log(x1)-log(x2);
gradf = [diff(f,x1);diff(f,x2)];
hessianf = hessian(f,[x1,x2]);
t = 2;
x1 = 1;
x2 = 2;
while true
    while true
            hessianf_inv = inv(hessianf);
            deltaX = 0.1*hessianf_inv*gradf;
            res = norm(deltaX,2);
            x1 = x1-deltaX(1,1);
            x2 = x2-deltaX(2,1);
        if res < 0.001
            break
        end
    end
    if 4.0/t < 0.001
        t
        x1
        x2
        break
    end
    t = 2*t;
end

function[x1,x2,res] = NewtonRaphson(f,x1,x2)

end



