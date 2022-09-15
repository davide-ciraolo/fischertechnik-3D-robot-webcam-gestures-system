function connectionFcn(src, ~)
if src.Connected
   message = read(src, src.NumBytesAvailable, "string")
   message = strtrim(message)
   global current_position;
   global old_position;
   global t;
   if strcmp(message, "open") == 1
     old_position = move_robot(t, old_position, 'rotation', current_position(1),'horizontal', current_position(2), 'vertical', current_position(3), 'gripper', gripper.open);
     current_position(4) = gripper.open;
   elseif strcmp(message, "close") == 1
     old_position = move_robot(t, old_position, 'rotation', current_position(1),'horizontal', current_position(2), 'vertical', current_position(3), 'gripper', gripper.close);
     current_position(4) = gripper.close;
   elseif strcmp(message, "forward") == 1
     old_position = move_robot(t, old_position, 'rotation', current_position(1),'horizontal', current_position(2) + 0.01, 'vertical', current_position(3), 'gripper', current_position(4));
     current_position(2) = current_position(2) + 0.01;
   elseif strcmp(message, "backward") == 1
     old_position = move_robot(t, old_position, 'rotation', current_position(1),'horizontal', current_position(2) - 0.01, 'vertical', current_position(3), 'gripper', current_position(4));
     current_position(2) = current_position(2) - 0.01;
   elseif strcmp(message, "left") == 1
     old_position = move_robot(t, old_position, 'rotation', current_position(1) - 10,'horizontal', current_position(2), 'vertical', current_position(3), 'gripper', current_position(4));
     current_position(1) = current_position(1) - 10;
   elseif strcmp(message, "right") == 1
     old_position = move_robot(t, old_position, 'rotation', current_position(1) + 10,'horizontal', current_position(2), 'vertical', current_position(3), 'gripper', current_position(4));
     current_position(1) = current_position(1) + 10;
   elseif strcmp(message, "up") == 1
     old_position = move_robot(t, old_position, 'rotation', current_position(1),'horizontal', current_position(2), 'vertical', current_position(3) + 0.01, 'gripper', current_position(4));
     current_position(3) = current_position(3) + 0.01;
   elseif strcmp(message, "down") == 1
     old_position = move_robot(t, old_position, 'rotation', current_position(1),'horizontal', current_position(2), 'vertical', current_position(3) - 0.01, 'gripper', current_position(4));
     current_position(3) = current_position(3) - 0.01;
   else
     disp("Nothing")
   end
else
   disp("Client has disconnected.")
end
end


