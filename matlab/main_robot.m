while true
    points =[42.19,0,0,gripper.open; 42.19,0.081,-0.06,gripper.open; 42.19,0.081,-0.06,gripper.close; 42.19,0.081,-0.018,gripper.close; 198.03,0.069,-0.018,gripper.close; 198.03,0.069,-0.06,gripper.close; 198.03,0.069,-0.06,gripper.open; 198.03,0,-0.06,gripper.open; -5,0,0,gripper.open];
    for i = 1:size(points,1)
        old_position = move_robot(t, old_position, 'rotation',points(i,1),'horizontal',points(i,2),'vertical',points(i,3),'gripper',points(i,4));
    end
    points =[198.03,0,-0.06,gripper.open; 198.03,0.069,-0.06,gripper.open; 198.03,0.069,-0.06,gripper.close; 198.03,0.069,0,gripper.close; 42.19,0.081,0,gripper.close; 42.19,0.081,-0.06,gripper.close; 42.19,0.081,-0.06,gripper.open; 42.19,0,-0.06,gripper.open; -5,0,0,gripper.open];
    for i = 1:size(points,1)
        old_position = move_robot(t, old_position, 'rotation',points(i,1),'horizontal',points(i,2),'vertical',points(i,3),'gripper',points(i,4));
    end
end