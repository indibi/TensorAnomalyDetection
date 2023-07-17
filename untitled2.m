
f = figure();
points=[];
points(1)=10;
step=1;
scatter(step,points(1),'b+');
hold on
while step <1000 & points(step)>0
    if randi(2)==1
        %win
        points(step+1) = ceil(points(step)/5) + points(step);
    else
        points(step+1) = points(step) - ceil(points(step)/5);
    end
    step = step+1;
    scatter(step,points(step),'b+');
end
