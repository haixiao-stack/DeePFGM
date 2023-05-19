readpath='D:\deepflare\DeePFGM\data\postdata\pmCDEF\pmCDEFarchives\pmD.stat';
savepath='D:\deepflare\DeePFGM\data\postdata';

d=0.72;

namelist=dir(strcat(readpath,'\*.Yave'));
for kk=1:7
    M=importdata(strcat(namelist(kk).folder,'\',namelist(kk).name));
    
    subplot(1,7,kk);
    title(strtok(namelist(kk).name,'.'));xlabel('T(K)');ylabel('radius(cm)'); hold on;
    plot(M.data(:,4),M.data(:,1)*d,'r*');hold on;  %Temperature of Exp-Renold average data

%     title(strtok(namelist(kk).name,'.'));xlabel('Z');ylabel('radius(cm)'); hold on;
%     plot(M.data(:,2),M.data(:,1)*d,'r*');hold on; %Mass Fraction of Exp-Renold average data

    %title(strtok(namelist(kk).name,'.'));xlabel('Zvar');ylabel('radius(cm)'); hold on;
    %plot(M.data(:,3),M.data(:,1)*d,'r*');hold on;  %Zvar of Exp-Renold average data

    %Expdata=cat()
end

namelist=dir(strcat(readpath,'\*.Yfav'));
for kk=1:7
    M=importdata(strcat(namelist(kk).folder,'\',namelist(kk).name));
    
    subplot(1,7,kk);
    plot(M.data(:,4),M.data(:,1)*d,'bo');hold on;  %Temperature of Exp-Renold average data

%     plot(M.data(:,2),M.data(:,1)*d,'bo');hold on; %Mass Fraction of Exp-Renold average data

    %plot(M.data(:,3),M.data(:,1)*d,'bo');hold on;  %Zvar of Exp-Renold average data
end



readpath_Sim='D:\deepflare\DeePFGM\data\postdata\mesh_initial_Tflamelet_RNGkE_modify_C1';
namelist=dir(strcat(readpath_Sim,'\*.csv'));
len=length(namelist);
for kk=1:len
    M=importdata(strcat(namelist(kk).folder,'\',namelist(kk).name));
    
    subplot(1,len,kk);
%     plot(M.data(:,41),M.data(:,58)*100,'b--');hold on;
%      plot(M.data(:,58),M.data(:,85)*100,'b-');hold on; 
    plot(M.data(:,2),M.data(:,20)*100,'b-');hold on;  %Temperature of Sim data

%        plot(M.data(:,6)*6.4056,M.data(:,20)*100,'b-');hold on; %Mass Fraction of Sim data
%     plot(M.data(:,67)*6.4056,M.data(:,85)*100,'b--');hold on;
    %plot(M.data(:,8),M.data(:,20)*100,'r');hold on;  %progress variable of Sim data
end

readpath_Sim='D:\deepflare\DeePFGM\data\postdata\output_T';
namelist=dir(strcat(readpath_Sim,'\*.csv'));
len=length(namelist);
for kk=1:len
    M=importdata(strcat(namelist(kk).folder,'\',namelist(kk).name));
    
    subplot(1,len,kk);
    plot(M.data(:,23),M.data(:,20)*100,'r-');hold on;  %Temperature of Sim data

%     plot(M.data(:,6)*6.4056,M.data(:,20)*100,'r');hold on; %Mass Fraction of Sim data
%     plot(M.data(:,45)*6.4056,M.data(:,58)*100,'r-');hold on;
%     plot(M.data(:,41),M.data(:,58)*100,'r--');hold on;  %progress variable of Sim data
end
legend('Exp-Renold average','Exp-Favre average','FGM','PaSR dynamicscale');
% legend('PaSR dynamicscale true','PaSR dynamicscale false');
% 'Exp-Renold average','Exp-Favre average',
% for kk=1:len
%    subplot(1,len,kk);
%    set(gca,'xlim',[0,0.15])
% end