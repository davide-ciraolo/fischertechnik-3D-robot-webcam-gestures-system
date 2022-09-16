[~,hostname] = system('hostname');
hostname = string(strtrim(hostname));
address = resolvehost(hostname,"address");
srv = tcpip(address, 7777, 'NetworkRole', 'server','BytesAvailableFcnMode','terminator');
srv.BytesAvailableFcn = @connectionFcn;
fopen(srv);
global current_position;
current_position = [0; 0; 0; gripper.open];