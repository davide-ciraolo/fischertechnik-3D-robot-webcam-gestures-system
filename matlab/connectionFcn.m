function connectionFcn(src, ~)
    global current_position;
    global old_position;
    global t;

    saved_positions = [];
    tmp_position = old_position;

    message = fscanf(src);
    message = strtrim(message);
    disp(message)

    if strcmp(message, "open") == 1
        tmp_position = move_robot(t, old_position, 'rotation', current_position(1),'horizontal', current_position(2), 'vertical', current_position(3), 'gripper', gripper.open);
        if ~isequal(tmp_position, old_position)
            old_position = tmp_position;
            current_position(4) = gripper.open;
        end
    elseif strcmp(message, "close") == 1
        tmp_position = move_robot(t, old_position, 'rotation', current_position(1),'horizontal', current_position(2), 'vertical', current_position(3), 'gripper', gripper.close);
        if ~isequal(tmp_position, old_position)
            old_position = tmp_position;
            current_position(4) = gripper.close;
        end
    elseif strcmp(message, "forward") == 1
        tmp_position = move_robot(t, old_position, 'rotation', current_position(1),'horizontal', current_position(2) + 0.01, 'vertical', current_position(3), 'gripper', current_position(4));
        if ~isequal(tmp_position, old_position)
            old_position = tmp_position;
            current_position(2) = current_position(2) + 0.01;
        end
    elseif strcmp(message, "backward") == 1
        tmp_position = move_robot(t, old_position, 'rotation', current_position(1),'horizontal', current_position(2) - 0.01, 'vertical', current_position(3), 'gripper', current_position(4));
        if ~isequal(tmp_position, old_position)
            old_position = tmp_position;
            current_position(2) = current_position(2) - 0.01;
        end
    elseif strcmp(message, "left") == 1
        tmp_position = move_robot(t, old_position, 'rotation', current_position(1) - 10,'horizontal', current_position(2), 'vertical', current_position(3), 'gripper', current_position(4));
        if ~isequal(tmp_position, old_position)
            old_position = tmp_position;
            current_position(1) = current_position(1) - 10;
        end
    elseif strcmp(message, "right") == 1
        tmp_position = move_robot(t, old_position, 'rotation', current_position(1) + 10,'horizontal', current_position(2), 'vertical', current_position(3), 'gripper', current_position(4));
        if ~isequal(tmp_position, old_position)
            old_position = tmp_position;
            current_position(1) = current_position(1) + 10;
        end
    elseif strcmp(message, "up") == 1
        tmp_position = move_robot(t, old_position, 'rotation', current_position(1),'horizontal', current_position(2), 'vertical', current_position(3) + 0.01, 'gripper', current_position(4));
        if ~isequal(tmp_position, old_position)
            old_position = tmp_position;
            current_position(3) = current_position(3) + 0.01;
        end
    elseif strcmp(message, "down") == 1
        tmp_position = move_robot(t, old_position, 'rotation', current_position(1),'horizontal', current_position(2), 'vertical', current_position(3) - 0.01, 'gripper', current_position(4));
        if ~isequal(tmp_position, old_position)
            old_position = tmp_position;
            current_position(3) = current_position(3) - 0.01;
        end
    elseif strcmp(message, "stop") == 1
        disp("The current position is: ");
        disp(current_position);
        saved_positions = [saved_positions; current_position];
        disp("Current saved positions: ");
        disp(saved_positions);
    else
        disp("Nothing");
    end

end


