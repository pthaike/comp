logdown = log(down);
x = logdown(1:end-30);
xmean = mean(down(end-60:end-30))*ones(30,1);
dlogdown = diff(x);
gddown = dlogdown;
%Md = arima(2,1,1);
Md = arima('Constant',0,'D',1,'Seasonality',7,'MALags',2,'SMALags',14);
Fit = estimate(Md,gddown);
[YF,YMSE] = forecast(Fit,29,'Y0',gddown);	

% back to prediction
pred = cumsum([xmean(1); YF]);

figure
h1 = plot(down,'Color',[.7,.7,.7]);
hold on
h2 = plot(155:184,pred,'b','LineWidth',2);
hold off

%predict
logdown = log(down);
x = logdown;
xmean = mean(down(end-30:end))*ones(60,1);
dlogdown = diff(x);
gddown = dlogdown;
%Md = arima(2,1,1);
Md = arima('Constant',0,'D',1,'Seasonality',7,'MALags',3,'SMALags',7);
Fit = estimate(Md,gddown);
[YF,YMSE] = forecast(Fit,60,'Y0',gddown);	

% back to prediction
pred = cumsum([xmean(1); YF])





d = down;
data = iddata(d);
sys = armax(data(1:end-30),[1,2]);
yp = predict(sys,data,30);
figure
h1 = plot(down,'Color',[.7,.7,.7]);
hold on
h2 = plot(154:183,get(yp),'b','LineWidth',2);







init_sys = idpoly([1 -0.99],[],[1 -1 0.2]);
e = iddata([],randn(400,1));
data = sim(init_sys,e);
na = 1;
nb = 2;
sys = armax(data(1:200),[na nb]);
K = 4;
yp = predict(sys,data,K);
plot(data(201:400),yp(201:400));
legend('Simulated data','Predicted data');