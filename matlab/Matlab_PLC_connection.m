global t;
global old_position;

old_position = [0;0;0;0;0;0;0;0];
t = tcpip('192.168.0.1', 2000, 'NetworkRole', 'server');
fopen(t);
