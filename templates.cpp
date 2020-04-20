#include <iostream>
#include <cstdio>
#include <algorithm>
#include <string>
#include<bits/stdc++.h>
#define print(a, i, j) for(int k=i;k<=j;k++)cout<<a[k]<<((k==j)?'\n':' ');
using namespace std;
typedef long long ll;


//素数筛
const int maxn = 1000000 + 10;
const int maxp = 700000;
int vis[maxn];
int prime[maxp];

void sieve(int n)
{
    int m = (int)sqrt(n+0.5);
    memset(vis,0,sizeof(vis));
    for(int i=2;i<=m;i++)
        if(!vis[i])
            for(int j=i*i;j<=n;j+=i)vis[j] = 1;
}
int gen_primes(int n)
{
    sieve(n);
    int c = 0;
    for(int i=2;i<=n;i++)if(!vis[i])
        prime[c++] = i;
    return c;
}

//gcd
ll gcd(ll a,ll b)
{
    return b==0?a:gcd(b,a%b);
}

void exgcd(ll a,ll b,ll &d,ll &x,ll &y)
{
    if(!b){
        d=a;x=1;y=0;
    }
    else {
        exgcd(b,a%b,d,y,x);
        y-=x*(a/b);
    }
}

//快速幂
ll mul_mod(ll a,ll b,int n)
{
    return a*b%n;
}
ll pow_mod(ll a,ll p,ll n)
{
    if(p==0)return 1;
    ll ans = pow_mod(a,p/2,n);
    ans=ans*ans%n;
    if(p%2==1)ans=ans*a%n;
    return ans;
}

//逆元
ll inv(ll a,ll n){

    ll d,x,y;
    exgcd(a,n,d,x,y);
    return d==1?(x+n)%n:-1;
}


//欧拉函数&欧拉打表
int euler_phi(int n)
{
    int m =(int)sqrt(n+0.5);
    int ans=n;
    for(int i=2;i<=m;i++)if(n%i==0){

        ans=ans/i*(i-1);
        while(n%i==0)n/=i;
    }
    if(n>1)ans=ans/n*(n-1);
    return ans;
}
int phi[maxn];
void phi_table(int n) {
    for(int i=2;i<=n;i++)phi[i]=0;
    phi[1]=1;
    for(int i=2;i<=n;i++)if(!phi[i])
        for(int j=i;j<=n;j+=i){

            if(!phi[j])phi[j]=j;
            phi[j]=phi[j]/i*(i-1);
        }
}

//并查集
const int maxn=100000+10;
int pa[maxn];
int findset(int x){return pa[x]!=x?pa[x]=findset(pa[x]):x;}

ll china(int n,int *a,int *m) {
    ll M = 1, d, y ,x = 0;
    for(int i=0;i<n;i++)M*=m[i];
    for(int i=0;i<n;i++){
        ll w = M/m[i];
        exgcd(m[i],w,d,d,y);
        x = (x+y*w*a[i])%M;
    }
    return (x+M)%M;
}

int log_mod(int a,int b,int n) {
    int m,v, e = 1, i;
    m = (int)sqrt(n+0.5);
    v = inv(pow_mod(a,m,n), n);
    map<int,int>x;
    x[1]=0;
    for(i=1;i<m;i++){
        e = mul_mod(e,a,n);
        if(!x.count(e))x[e] = i;
    }
    for(i=0;i<m;i++){
        if(x.count(b))return i*m+x[b];
        b = mul_mod(b,v,n);
    }
    return -1;
}


#define lowbit(x) x&-x
//树状数组
int sum(int x){
    int ret = 0;
    whiel(x>0){
        ret += C[x];
        x -= lowbit(x);
    }
    return ret;
}
void add(int x,int d){
    while(c<=n){
        C[x] += d;
        x += lowbit(x);
    }
}

//RMQ ST算法
void RMQ_init(const vector<int> &A){
    int n= A.size();
    for(int i=0;i<n;i++)d[i][0] = A[i];
    for(int i=0;i<n;i++)
    for(int j=i;(1<<j)<=n;j++)
        for(int i=0;i+(1<<j)-1<n;i++)
            d[i][j] = min(d[i][j-1],d[i+(1<<(j-1))][j-1]);
}
int RMQ(int L,int R){//查询区间从零开始计数的
    int k=0;
    while((1<<(k+1)) <= R-L+1)k++;
    return min(d[L][k], d[R-(1<<k)+1][k]);
}




//线段树,单点修改，查询区间最小值
//用法o:1  整个区间：1,n
int ql,qr;
int query(int o,int L,int R){
    int M = L + (R-L)/2, ans = INF;
    if(ql <= L && R <= qr)return minv[o];
    if(ql <= M)ans = min(ans, query(o*2, L , M));
    if(M < qr)ans = min(ans, query(o*2+1, M+1, R));
    return ans;
}
int p,v;//修改A[p] = v;
void update(int o,int L,int R){
    int M = L + (R-L)/2;
    if(L==R)minv[o] = v;
    else {
        if(p <= M)update(o*2,L, M);else update(o*2+1,M+1,R);
        minv[o] = min(minv[o*2],minv[o*2+1]);
    }
}



//线段树,区间修改(区间add),查询区间和/区间最小值/区间最大值
//query之间记得给_sum,_min,_max赋初始值
int y1,y2;int v;
void maintain(int o,int L,int R){
    int lc = o*2,rc = o*2+1;
    sumv[o] = minv[o] = maxv[o] = 0;
    if(R>L){
        sumv[o] = sumv[lc] + sumv[rc];
        minv[o] = min(minv[lc],minv[rc]);
        maxv[o] = max(maxv[lc],maxv[rc]);
    }
    minv[o] += addv[o];maxv[o] += addv[o]; sumv[o] +=addv[o] * (R-L+1);
}

void update(int o,int L,int R){
    int lc = o*2,rc = o*2+1;
    if(y1<=L && y2 >= R){
        addv[o] += v;
    }else {
        int M = L + (R-L)/2;
        if(y1 <= M)update(lc, L ,M);
        if(y2 > M)update(rc, M+1,R);
    }
    maintain(o,L,R);
}

int _min,_max,_sum;
void query(int o,int L, int R, int add){
    if(y1<=L && y2 >=R){
        _sum +=sumv[o] + add*(R-L+1);
        _min=min(_min,minv[o]+add);
        _max=max(_max,maxv[o]+add);
    }else {
        int M = L + (R-L)/2;
        if(y1<=M)query(o*2,L,M,add+addv[o]);
        if(y2>M)query(o*2+1,M+1,R,add+addv[o]);
    }
    
}




//线段树，区间修改(区间set),查询区间和/区间最小值/区间最大值
//query之前记得给_sum,_min,_max赋初始值
//如果会修改为负数，则需要修改一下pushdown
int sumv[10005];
int minv[10005];
int maxv[10005];
int setv[10005];
int yy1,yy2,v;
int _max,_min,_sum;
void maintain(int o,int L,int R){
    int lc = o*2,rc = o*2+1;
    sumv[o] = minv[o] = maxv[o] = 0;
    if(R>L){
        sumv[o] = sumv[lc] + sumv[rc];
        minv[o] = min(minv[lc],minv[rc]);
        maxv[o] = max(maxv[lc],maxv[rc]);
    }
    if(setv[o]==-1)return;
    minv[o] = setv[o];maxv[o] = setv[o]; sumv[o] = setv[o] * (R-L+1);
}
void pushdown(int o){
    int lc = o*2,rc = o*2+1;
    if(setv[o]>=0){
        setv[lc] = setv[rc] = setv[o];
        setv[o] = -1;
    }
}
void update(int o,int L,int R){
    int lc = o*2,rc = o*2+1;
    if(yy1<=L&&yy2>=R){
        setv[o] = v;
    }else {
        pushdown(o);
        int M = L + (R-L)/2;
        if(yy1<=M)update(lc,L,M);else maintain(lc,L,M);
        if(yy2>M)update(rc,M+1,R);else maintain(rc,M+1,R);
    }
    maintain(o,L,R);
}
void query(int o,int L,int R){
    if(setv[o]>=0){
        _sum+=setv[o]*(min(R,yy2)-max(L,yy1)+1);
        _min=min(_min,setv[o]);
        _max=max(_max,setv[o]);
    }else if(yy1<=L&&yy2>=R){
        _sum+=sumv[o];
        _min=min(_min,minv[o]);
        _max=max(_max,maxv[o]);
    }else{
        int M = L + (R-L)/2;
        if(yy1<=M)query(o*2,L,M);
        if(yy2>M)query(o*2+1,M+1,R);
    }
}
