\\ n is a positive integer.
\\ q is a prime power.
order(n,q)=prod(i=0,n-1,q^n-q^i)/(q-1)
sl(N=1000)=for(n=2,10,for(q=2,10000,if(isprimepower(q),if(order(n,q)<N,print(n," ",factor(q)," ",order(n,q)),),)))
