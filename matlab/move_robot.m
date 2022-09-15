function old_position = move_robot(connection, old_position, varargin)
    connection.timeout = 180;
    nreqargs = 2;
    bit16rotation = [old_position(2,1) old_position(1,1)];
    bit16vertical = [old_position(4,1) old_position(3,1)];
    bit16horizontal = [old_position(6,1) old_position(5,1)];
    gripper = old_position(7,1);
    
    reached_rotation = typecast(uint8(bit16rotation),'uint16');
    reached_vertical = typecast(uint8(bit16vertical),'uint16');
    reached_horizontal = typecast(uint8(bit16horizontal),'uint16');
    
    rotation = typecast(uint8(bit16rotation),'uint16');
    vertical = typecast(uint8(bit16vertical),'uint16');
    horizontal = typecast(uint8(bit16horizontal),'uint16');
    
    if nargin > nreqargs
        i=1;
        while(i <= size(varargin,2))
            switch lower(varargin{i})
                case  'rotation'
                    temp_rotation = varargin{i+1};
                    rotation = round((temp_rotation+5)*9.85);
                    if rotation > 3300
                        rotation = 3300;
                    end
                    if rotation < 0
                        rotation = 0;
                    end
                    bit16rotation = typecast(uint16(rotation),'uint8');
                case 'horizontal'
                    temp_horizontal = varargin{i+1};
                    horizontal = round(temp_horizontal*821);
                    if horizontal > 78
                        horizontal = 78;
                    end
                    if horizontal < 0
                        horizontal = 0;
                    end
                    bit16horizontal = typecast(uint16(horizontal),'uint8');
                case 'vertical'
                    temp_vertical = varargin{i+1};
                    vertical = round(temp_vertical*(-12857));
                    
                    if vertical > 1800
                        vertical = 1800;
                    end
                    if vertical < 0
                        vertical = 0;
                    end
                    bit16vertical = typecast(uint16(vertical),'uint8');
                case 'gripper'
                    gripper = varargin{i+1};
            end
            i= i+1;
        end
    end
    
    codified_var = [bit16rotation(2); bit16rotation(1); bit16vertical(2); bit16vertical(1); bit16horizontal(2); bit16horizontal(1); gripper; 0];
    
    if (isequal(gripper, old_position(7)) && isequal(reached_horizontal, horizontal) && (reached_rotation >= rotation - 10 && reached_rotation <= rotation + 10 ) && (reached_vertical >= vertical - 10 && reached_vertical <= vertical + 10))
        return
    end
    
    fwrite(connection, codified_var)
    old_position = fread(connection,8);
end

