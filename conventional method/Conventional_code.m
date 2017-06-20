
%% Data load
clear all; clc;

for block=1:27
 name={'B0101T','B0102T','B0103T','B0201T','B0202T','B0203T','B0301T','B0302T','B0303T','B0401T','B0402T','B0403T','B0501T','B0502T','B0503T','B0601T','B0602T','B0603T','B0701T','B0702T','B0703T','B0801T','B0802T','B0803T','B0901T','B0902T','B0903T'};
    
    file3 = ['C:\Users\Oyeon Kwon\Desktop\bbci_IV\BCICIV_2b_mat\',name{block},'\'];
    filename10 = ['s.mat'];   
    filename11 = ['h.mat'];  
    ss = load([file3, filename10]);     
    hh = load([file3, filename11]);  

h=hh.h;
s=ss.s;

marker={'1','left';'2','right';};

CNT.x=s;
CNT.t=h.TRIG';
CNT.fs=h.SampleRate;
CNT.y_dec=h.Classlabel';

idx1=CNT.y_dec==1;
idx2=CNT.y_dec==2;
CNT.y_logic(1,:)=idx1;
CNT.y_logic(2,:)=idx2;

lable_1=find(CNT.y_dec==1);
CNT.y_class(lable_1)={'left'};
lable_2=find(CNT.y_dec==2);
CNT.y_class(lable_2)={'right'};

CNT.class=marker;
CNT.chan= {'C3','Cz','C4','EOGl','EOGc','EOGr'};
CNT=prep_selectChannels(CNT,{'Name',{'C3','Cz','C4'}});
CNT=prep_selectClass(CNT,{'class',{'left','right'}});
real_cnt{1,block}=CNT;

CNTfilt=prep_filter(CNT, {'frequency',[8 30];'fs',250});
SMT=prep_segmentation(CNTfilt, {'interval', [4000 7000]});

cnt{1,block}=SMT;

clear Converting CNT idx1 idx2 lable_1 lable_2 s ss h hh 
end
% 
% ch_3_cnt=stft(s(:,1),32,,500,250);

%% Train Test SMT
a=1;
for num=0:3:24
    for iter=1:3
        new_cnt{a,iter}=cnt{1,num+iter};
    end
    a=a+1;
end

% for nu=1:9
% for num=1:3
%   sub(1,num)=new_cnt{nu,num};
% end
% sig{:,:,:,nu}= [sub(1).x sub(2).x sub(3).x];
% label{:,:,nu}= [sub(1).y_dec sub(2).y_dec sub(3).y_dec];
% end

% train (90%) test (10%)
for sub=1:9
sub_cnt = [new_cnt{sub,1} new_cnt{sub,2} new_cnt{sub,3}];

% converting
converting.t=[sub_cnt(1).t sub_cnt(2).t sub_cnt(3).t];
converting.fs=[sub_cnt(1).fs];
converting.y_dec=[sub_cnt(1).y_dec sub_cnt(2).y_dec sub_cnt(3).y_dec];
converting.y_logic=[sub_cnt(1).y_logic sub_cnt(2).y_logic sub_cnt(3).y_logic];
converting.y_class=[sub_cnt(1).y_class sub_cnt(2).y_class sub_cnt(3).y_class];
converting.class=[sub_cnt(1).class];
converting.chan=[sub_cnt(1).chan];
converting.x=[sub_cnt(1).x sub_cnt(2).x sub_cnt(3).x];
converting.ival=[sub_cnt(1).ival];

smt{sub}=converting;
c(sub)=converting;
end
%%

for n=1:9
SMT=smt{1,n};
% SMT = prep_envelope(SMT);
class1data{1,n}=SMT.x(:,find(SMT.y_logic(1,:)),:);
class2data{1,n}=SMT.x(:,find(SMT.y_logic(2,:)),:);
end

conca_class1= [class1data{1,1} class1data{1,2} class1data{1,3} class1data{1,4} class1data{1,5} class1data{1,6} class1data{1,7} class1data{1,8} class1data{1,9}];
conca_class2= [class2data{1,1} class2data{1,2} class2data{1,3} class2data{1,4} class2data{1,5} class2data{1,6} class2data{1,7} class2data{1,8} class2data{1,9}];

mean_class1=mean(conca_class1,2);
mean_class2=mean(conca_class2,2);

% mean_class1=reshape(mean_class1,751,3,1);
% mean_class2=reshape(mean_class2,751,3,1);
%%
% mu_class1 = [VarName1 VarName2 VarName3];
% mu_class2 = right;
% dat.x=mean_class2;
% 
% s = size(dat.x);
% dat.x= reshape(abs(hilbert(dat.x(:,:))),s);
% 
% [t,~] = size(dat.x);
% n = round(100*250/1000);
% x = zeros(size(dat.x));
% 
%         for i=1:min(n,t)
%             x(i,:) = mean(dat.x([1:i],:),1);
%         end
%         for i=n+1:t
%             x(i,:) = mean(dat.x([i-n+1:i],:),1);
%         end
% x=reshape(x,751,3,1);
%%
figure
plot(mean_class1(:,1));
title('C3');
% figure();
hold on
plot(mean_class2(:,1));
legend('left','right');

figure
plot(mean_class1(:,3));
title('C4');
% figure();
hold on
plot(mean_class2(:,3));
legend('left','right');

%% Log-var & LDA 
% 10-fold-Crossvalidation with 10 iter /// C3 & C4 & Cz

for num=1:9
for iter=1:10
SMT= smt{1,num};
% CV.var.band=[7 30];
% CV.var.interval=[750 3500];
CV.prep={ % commoly applied to training and test data before data split
%     'CNT=prep_filter(CNT, {"frequency", band})'
%     'SMT=prep_segmentation(CNT, {"interval", interval})'
       'SMT= prep_selectTrials(SMT,{"Index",[1:400]})'
    };
CV.train={
        '[SMT, CSP_W, CSP_D]=func_csp(SMT,{"nPatterns", [3]})'
    'FT=func_featureExtraction(SMT, {"feature","logvar"})'
    '[CF_PARAM]=func_train(FT,{"classifier","LDA"})'
    };
CV.test={
        'SMT=func_projection(SMT, CSP_W)'
    'FT=func_featureExtraction(SMT, {"feature","logvar"})'
    '[cf_out]=func_predict(FT, CF_PARAM)'
    };
CV.option={
'KFold','10'
% 'leaveout'
};

[loss]=eval_crossValidation(SMT, CV); % input : eeg, or eeg_epo
result= 1-loss';
aaa(1,iter)=result';
cv_result=aaa';
end
acc(:,num)=cv_result;
end
accuracy=mean(acc,1);
final=accuracy';



%% Log-var Feature distribution 

% based on "GOOD" subject
SMT= smt{1,4};
SMT= prep_selectTrials(SMT,{'Index',[1:400]});
SMT=prep_selectChannels(SMT,{'Name',{'C3','C4'}});
FT=func_featureExtraction(SMT, {'feature','logvar'});
trainingData = [FT.x' FT.y_dec']; % Se1 vs Se2
%dividing data into two classes (and removing the label)
ldaParams.classLabels = sort(unique(trainingData(:,end)));
lot_class1Data_s1 = trainingData(trainingData(:,end)==ldaParams.classLabels(1),1:(end-1));
lot_class2Data_s1 = trainingData(trainingData(:,end)==ldaParams.classLabels(2),1:(end-1));

xTr=FT.x;
yTr=FT.y_logic;
nClasses= size(yTr,1);

d= size(xTr, 1);
X= zeros(d,0);
C_mean_s1= zeros(d, nClasses);

for ci= 1:nClasses,
  idx= find(yTr(ci,:));
  C_mean_s1(:,ci)= mean(xTr(:,idx),2);
  X= [X, xTr(:,idx) - C_mean_s1(:,ci)*ones(1,length(idx))];
end

class1data=X(:,find(yTr(1,:)));
class2data=X(:,find(yTr(2,:)));

%%% bbci covariance.. 
[C_cov_s1, C_s1.gamma]= clsutil_shrinkage(X);
[C_cov1_s1, C_s1.gamma1]= clsutil_shrinkage(class1data);
[C_cov2_s1, C_s1.gamma2]= clsutil_shrinkage(class2data);
C_invcov_s1= pinv(C_cov_s1);
C_invcov1= pinv(C_cov1_s1);
C_invcov2= pinv(C_cov2_s1);

%%% bbci weight and bias
C_s1.w= C_invcov_s1*C_mean_s1;
C_s1.b= -0.5*sum(C_mean_s1.*C_s1.w,1)';

if nClasses==2
  C_s1.w= C_s1.w(:,2) - C_s1.w(:,1);
  C_s1.b= C_s1.b(2)-C_s1.b(1);
end

tr_forplot1.mu1=C_mean_s1(:,1); % 2*2 ... 첫번째 col (class1) / 두번째 col (class 2)
tr_forplot2.mu2=C_mean_s1(:,2);
tr_forplot1.sigma1=C_cov1_s1;
tr_forplot2.sigma2=C_cov2_s1;
tr_forplot1.class1Data=class1data;
tr_forplot2.class2Data=class2data;

figure
% Session 1 data "GREEN"
hold on;
h1 = plot_gaussian_ellipsoid([tr_forplot1.mu1], [tr_forplot1.sigma1]);
set(h1,'color','b'); 
hold on;
h2 = plot_gaussian_ellipsoid([tr_forplot2.mu2], [tr_forplot2.sigma2]);
set(h2,'color','m'); 
hold on;

hold on;
scatter(lot_class1Data_s1(:,1), lot_class1Data_s1(:,2),'Marker','+','MarkerEdgeColor','b');
hold on;
scatter(lot_class2Data_s1(:,1), lot_class2Data_s1(:,2),'Marker','+','MarkerEdgeColor','m');
% legend('training');
hold on;

% slope 
C_s1.w;
C_s1.b;
% hold on
% f = @(slope,bias,dat) ((-1/slope) *dat + bias*ones(size(dat)));
% x = linspace(-4,6,4);
% y = f(C_s1.w(2)/C_s1.w(1), C_s1.b, x);
% plot(x,y,'r')
% xlim([0 4])
% ylim([0 4])

xlabel('Channel C3');
ylabel('Channel C4');
title('Feature distribution');

legend('Class1 dist','Class2 dist','Class1 feature','Class2 feature','decision line','Location','southeast');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% based on "BAD" subject
SMT= smt{1,3};
SMT=prep_selectChannels(SMT,{'Name',{'C3','C4'}});
FT=func_featureExtraction(SMT, {'feature','logvar'});
trainingData = [FT.x' FT.y_dec']; % Se1 vs Se2
%dividing data into two classes (and removing the label)
ldaParams.classLabels = sort(unique(trainingData(:,end)));
lot_class1Data_s1 = trainingData(trainingData(:,end)==ldaParams.classLabels(1),1:(end-1));
lot_class2Data_s1 = trainingData(trainingData(:,end)==ldaParams.classLabels(2),1:(end-1));

xTr=FT.x;
yTr=FT.y_logic;
nClasses= size(yTr,1);

d= size(xTr, 1);
X= zeros(d,0);
C_mean_s1= zeros(d, nClasses);

for ci= 1:nClasses,
  idx= find(yTr(ci,:));
  C_mean_s1(:,ci)= mean(xTr(:,idx),2);
  X= [X, xTr(:,idx) - C_mean_s1(:,ci)*ones(1,length(idx))];
end

class1data=X(:,find(yTr(1,:)));
class2data=X(:,find(yTr(2,:)));

%%% bbci covariance.. 
[C_cov_s1, C_s1.gamma]= clsutil_shrinkage(X);
[C_cov1_s1, C_s1.gamma1]= clsutil_shrinkage(class1data);
[C_cov2_s1, C_s1.gamma2]= clsutil_shrinkage(class2data);
C_invcov_s1= pinv(C_cov_s1);
C_invcov1= pinv(C_cov1_s1);
C_invcov2= pinv(C_cov2_s1);

%%% bbci weight and bias
C_s1.w= C_invcov_s1*C_mean_s1;
C_s1.b= -0.5*sum(C_mean_s1.*C_s1.w,1)';

if nClasses==2
  C_s1.w= C_s1.w(:,2) - C_s1.w(:,1);
  C_s1.b= C_s1.b(2)-C_s1.b(1);
end

tr_forplot1.mu1=C_mean_s1(:,1); % 2*2 ... 첫번째 col (class1) / 두번째 col (class 2)
tr_forplot2.mu2=C_mean_s1(:,2);
tr_forplot1.sigma1=C_cov1_s1;
tr_forplot2.sigma2=C_cov2_s1;
tr_forplot1.class1Data=class1data;
tr_forplot2.class2Data=class2data;

figure
% Session 1 data "GREEN"
hold on;
h1 = plot_gaussian_ellipsoid([tr_forplot1.mu1], [tr_forplot1.sigma1]);
set(h1,'color','b'); 
hold on;
h2 = plot_gaussian_ellipsoid([tr_forplot2.mu2], [tr_forplot2.sigma2]);
set(h2,'color','m'); 
hold on;

hold on;
scatter(lot_class1Data_s1(:,1), lot_class1Data_s1(:,2),'Marker','+','MarkerEdgeColor','b');
hold on;
scatter(lot_class2Data_s1(:,1), lot_class2Data_s1(:,2),'Marker','+','MarkerEdgeColor','m');
% legend('training');
hold on;

% slope 
C_s1.w;
C_s1.b;
% hold on
% f = @(slope,bias,dat) ((-1/slope) *dat + bias*ones(size(dat)));
% x = linspace(-4,6,4);
% y = f(C_s1.w(2)/C_s1.w(1), C_s1.b, x);
% plot(x,y,'r')
% xlim([0 4])
% ylim([0 4])

xlabel('Channel C3');
ylabel('Channel C4');
title('Feature distribution');

legend('Class1 dist','Class2 dist','Class1 feature','Class2 feature','decision line','Location','southeast');

%% CSP Feature distribution 

% based on "GOOD" subject
SMT= smt{1,4};
SMT=prep_selectChannels(SMT,{'Name',{'C3','C4'}});
[SMT, CSP_W, CSP_D]=func_csp(SMT,{'nPatterns', [1]});       %% CSP
FT=func_featureExtraction(SMT, {'feature','logvar'});
trainingData = [FT.x' FT.y_dec']; % Se1 vs Se2
%dividing data into two classes (and removing the label)
ldaParams.classLabels = sort(unique(trainingData(:,end)));
lot_class1Data_s1 = trainingData(trainingData(:,end)==ldaParams.classLabels(1),1:(end-1));
lot_class2Data_s1 = trainingData(trainingData(:,end)==ldaParams.classLabels(2),1:(end-1));

xTr=FT.x;
yTr=FT.y_logic;
nClasses= size(yTr,1);

d= size(xTr, 1);
X= zeros(d,0);
C_mean_s1= zeros(d, nClasses);

for ci= 1:nClasses,
  idx= find(yTr(ci,:));
  C_mean_s1(:,ci)= mean(xTr(:,idx),2);
  X= [X, xTr(:,idx) - C_mean_s1(:,ci)*ones(1,length(idx))];
end

class1data=X(:,find(yTr(1,:)));
class2data=X(:,find(yTr(2,:)));

%%% bbci covariance.. 
[C_cov_s1, C_s1.gamma]= clsutil_shrinkage(X);
[C_cov1_s1, C_s1.gamma1]= clsutil_shrinkage(class1data);
[C_cov2_s1, C_s1.gamma2]= clsutil_shrinkage(class2data);
C_invcov_s1= pinv(C_cov_s1);
C_invcov1= pinv(C_cov1_s1);
C_invcov2= pinv(C_cov2_s1);

%%% bbci weight and bias
C_s1.w= C_invcov_s1*C_mean_s1;
C_s1.b= -0.5*sum(C_mean_s1.*C_s1.w,1)';

if nClasses==2
  C_s1.w= C_s1.w(:,2) - C_s1.w(:,1);
  C_s1.b= C_s1.b(2)-C_s1.b(1);
end

tr_forplot1.mu1=C_mean_s1(:,1); % 2*2 ... 첫번째 col (class1) / 두번째 col (class 2)
tr_forplot2.mu2=C_mean_s1(:,2);
tr_forplot1.sigma1=C_cov1_s1;
tr_forplot2.sigma2=C_cov2_s1;
tr_forplot1.class1Data=class1data;
tr_forplot2.class2Data=class2data;

figure
% Session 1 data "GREEN"
hold on;
h1 = plot_gaussian_ellipsoid([tr_forplot1.mu1], [tr_forplot1.sigma1]);
set(h1,'color','b'); 
hold on;
h2 = plot_gaussian_ellipsoid([tr_forplot2.mu2], [tr_forplot2.sigma2]);
set(h2,'color','m'); 
hold on;

hold on;
scatter(lot_class1Data_s1(:,1), lot_class1Data_s1(:,2),'Marker','+','MarkerEdgeColor','b');
hold on;
scatter(lot_class2Data_s1(:,1), lot_class2Data_s1(:,2),'Marker','+','MarkerEdgeColor','m');
% legend('training');
hold on;

% slope 
C_s1.w;
C_s1.b;
hold on
f = @(slope,bias,dat) ((-1/slope) *dat + bias*ones(size(dat)));
x = linspace(-3,1,2);
y = f(C_s1.w(2)/C_s1.w(1), C_s1.b, x);
plot(x,y,'r')
% xlim([0 4])
% ylim([0 4])

xlabel('Channel C3');
ylabel('Channel C4');
title('Feature distribution');

legend('Class1 dist','Class2 dist','Class1 feature','Class2 feature','decision line','Location','southeast');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% based on "BAD" subject
SMT= smt{1,3};
SMT=prep_selectChannels(SMT,{'Name',{'C3','C4'}});
[SMT, CSP_W, CSP_D]=func_csp(SMT,{'nPatterns', [1]});   %% CSP
FT=func_featureExtraction(SMT, {'feature','logvar'});
trainingData = [FT.x' FT.y_dec']; % Se1 vs Se2
%dividing data into two classes (and removing the label)
ldaParams.classLabels = sort(unique(trainingData(:,end)));
lot_class1Data_s1 = trainingData(trainingData(:,end)==ldaParams.classLabels(1),1:(end-1));
lot_class2Data_s1 = trainingData(trainingData(:,end)==ldaParams.classLabels(2),1:(end-1));

xTr=FT.x;
yTr=FT.y_logic;
nClasses= size(yTr,1);

d= size(xTr, 1);
X= zeros(d,0);
C_mean_s1= zeros(d, nClasses);

for ci= 1:nClasses,
  idx= find(yTr(ci,:));
  C_mean_s1(:,ci)= mean(xTr(:,idx),2);
  X= [X, xTr(:,idx) - C_mean_s1(:,ci)*ones(1,length(idx))];
end

class1data=X(:,find(yTr(1,:)));
class2data=X(:,find(yTr(2,:)));

%%% bbci covariance.. 
[C_cov_s1, C_s1.gamma]= clsutil_shrinkage(X);
[C_cov1_s1, C_s1.gamma1]= clsutil_shrinkage(class1data);
[C_cov2_s1, C_s1.gamma2]= clsutil_shrinkage(class2data);
C_invcov_s1= pinv(C_cov_s1);
C_invcov1= pinv(C_cov1_s1);
C_invcov2= pinv(C_cov2_s1);

%%% bbci weight and bias
C_s1.w= C_invcov_s1*C_mean_s1;
C_s1.b= -0.5*sum(C_mean_s1.*C_s1.w,1)';

if nClasses==2
  C_s1.w= C_s1.w(:,2) - C_s1.w(:,1);
  C_s1.b= C_s1.b(2)-C_s1.b(1);
end

tr_forplot1.mu1=C_mean_s1(:,1); % 2*2 ... 첫번째 col (class1) / 두번째 col (class 2)
tr_forplot2.mu2=C_mean_s1(:,2);
tr_forplot1.sigma1=C_cov1_s1;
tr_forplot2.sigma2=C_cov2_s1;
tr_forplot1.class1Data=class1data;
tr_forplot2.class2Data=class2data;

figure
% Session 1 data "GREEN"
hold on;
h1 = plot_gaussian_ellipsoid([tr_forplot1.mu1], [tr_forplot1.sigma1]);
set(h1,'color','b'); 
hold on;
h2 = plot_gaussian_ellipsoid([tr_forplot2.mu2], [tr_forplot2.sigma2]);
set(h2,'color','m'); 
hold on;

hold on;
scatter(lot_class1Data_s1(:,1), lot_class1Data_s1(:,2),'Marker','+','MarkerEdgeColor','b');
hold on;
scatter(lot_class2Data_s1(:,1), lot_class2Data_s1(:,2),'Marker','+','MarkerEdgeColor','m');
% legend('training');
hold on;

% slope 
C_s1.w;
C_s1.b;
hold on
f = @(slope,bias,dat) ((-1/slope) *dat + bias*ones(size(dat)));
x = linspace(-3,1,2);
y = f(C_s1.w(2)/C_s1.w(1), C_s1.b, x);
plot(x,y,'r')
% xlim([0 4])
% ylim([0 4])

xlabel('Channel C3');
ylabel('Channel C4');
title('Feature distribution');

legend('Class1 dist','Class2 dist','Class1 feature','Class2 feature','decision line','Location','southeast');



