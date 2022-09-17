%[~,hostname] = system('hostname');
%hostname = string(strtrim(hostname));
%address = resolvehost(hostname,"address");
srv = tcpip('192.168.43.215', 7777, 'NetworkRole', 'server','BytesAvailableFcnMode','terminator');
srv.BytesAvailableFcn = @connectionFcn;
fopen(srv);
global current_position;
current_position = [0; 0; 0; gripper.open];