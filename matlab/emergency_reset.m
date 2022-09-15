function old_position = emergency_reset(connection)
    fclose(connection);
    old_position = [0;0;0;0;0;0;0;0];
    fopen(connection);