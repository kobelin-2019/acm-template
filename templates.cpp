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

//矩阵快速幂
struct Mat { ///结构体，矩阵类型
    int m[M][M];
} res,e;
void init_e(){
    //整数快速幂默认的ans是1，矩阵的话ans应为单位矩阵
    for(int i=1;i<=n;i++) {
        for(int j=1;j<=n;j++) {
            if(i==j)
                e.m[i][j]=1;
            else
                e.m[i][j]=0;
        }
    }
}
Mat Mul(Mat a,Mat b,int n) {
    Mat tmp;//定义一个临时的矩阵，存放A*B的结果
    for(int i = 1; i <= n; i++) {
        for(int j = 1; j <= n; j++) {
            tmp.m[i][j] = 0;
        }
    }
    for(itn i=1; i <= n; i++) {
        for(int j = 1; j <= n; j++) {
            for(int k = 1; k <= n; k++) {
                tmp.m[i][j] += a.m[i][k]*b.m[k][j];
            }
        }
    }
    return tmp;
}
///矩阵快速幂，求矩阵res的N次幂
Mat Mat_qpower(Mat base,int K){
    res=e;
    while(K) {
        if(K&1)
            res=Mul(res,base);
        base=Mul(base,base);
        K=K>>1;
    }
    return res;
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



//Trie字典树,未对拍测试
struct Trie{
    int ch[maxnode][sigma_size];
    int val[maxnode];
    int sz;
    Trie(){
        sz = 1;memset(ch[0],0,sizeof(ch[0]));
    }
    int idx(char c){
        return c-'a';
    }

    void insert(char *s,int v){
        int u = 0, n = strlen(s);
        for(int i=0;i<n;i++){
            int c=idx(s[i]);
            if(!ch[u][c]){
                memset(ch[sz],0,sizeof(ch[sz]));
                val[sz] = 0;
                ch[u][c] = sz++;
            }
            u = ch[u][c];
        }
        val[u] = v;
    }
};

//KMP字符串匹配，未对拍测试
void getFail(char *P,int *f){
    int m = strlen(P);
    f[0] = 0;f[1] =0;
    for(int i=1;i<m;i++){
        int j= f[i];
        while(j&&P[i]!=P[j])j=f[j];
        f[i+1]=P[i]==P[j]?j+1:0;
    }
}
void find(char *T,char *P,int *f){
    int n = strlen(T), m = strlen(P);
    getFail(P,f);
    int j = 0;
    for(int i=0;i<n;i++){
        while(j&&P[j]!=T[i])j=f[j];
        if(P[j]==T[i])j++;
        if(j==m)printf("%d\n",i-m+1);
    }
}

//AC自动机
//hdu 2222在主串中统计模式串们出现的总次数

const int maxn = 1000000 + 100;
int n,ans;
const int SIGMA_SIZE = 51;
const int maxnode = 1000000 + 100;

bool vis[maxn];
map<string, int >ms;
int ch[maxnode][SIGMA_SIZE+5];
int val[maxnode];
int idx(char c){
    return c - 'a';
}
struct Trie{
    int sz;
    Trie(){
        sz = 1;memset(ch[0],0,sizeof(ch[0]));memset(vis,0,sizeof(vis));
    }
    void insert(char *s){
        int u = 0, n = strlen(s);
        for(int i=0;i<n;i++){
            int c = idx(s[i]);
            if(!ch[u][c]){
                memset(ch[sz],0,sizeof(ch[sz]));
                ch[u][c] = sz++;
            }
            u = ch[u][c];
        }
        val[u]++;
    }
};
int last[maxn],f[maxn];
void print(int j){
    if(j&&!vis[j]){
        ans +=val[j];vis[j] = 1;
        print(last[j]);
    }
}
void getFail(){
    queue<int>q;
    f[0] = 0;
    for(int c = 0;c<SIGMA_SIZE;c++){
        int u = ch[0][c];
        if(u){
            f[u] = 0;q.push(u);last[u] = 0;
        }
    }

    while(!q.empty()){
        int r=q.front();q.pop();
        for(int c = 0;c<SIGMA_SIZE;c++){
            int u=ch[r][c];
            if(!u){
                ch[r][c] = ch[f[r]][c];
                continue;
            }
            q.push(u);
            int v = f[r];
            while(v&&!ch[v][c])v = f[v];
            f[u] = ch[v][c];
            last[u] = val[f[u]]?f[u]:last[f[u]];
        }
    }
}

void find_T(char * T){
    int n = strlen(T);
    int j = 0;
    for(int i=0;i<n;i++){
        int c = idx(T[i]);
        j = ch[j][c];
        if(val[j])print(j);
        else if(last[j])print(last[j]);
    }
}
char tmp[105];
char text[1000000+1000];

int main()
{
    //freopen("in.txt","r",stdin);
    int T;
    cin>>T;
    while(T--){
        scanf("%d",&n);
        Trie trie;
        memset(val,0,sizeof(val));
        ans = 0;
        for(int i=0;i<n;i++){
            scanf("%s",tmp);
            trie.insert(tmp);
        }
        getFail();
        scanf("%s",text);
        find_T(text);
        cout<<ans<<endl;
    }



    return 0;
}


//字典序比较模板
template<class T>
bool lexicographicallySmaller(vector<T> a, vector<T> b){
    int n = a.size();
    int m = b.size();
    int i;
    for(i = 0;i<n&&i<m;i++){
        if(a[i]<b[i])return true;
        else if(b[i]<a[i])return false;
    }
    return (i==n&&i<m);
}

//后缀数组1,能求出sa, rk, 复杂度O(n(logn)^2),容易理解
const int maxn = 200005;
char s[maxn];
int sa[maxn];
int rk[maxn];
int tmp[maxn+1];
int n,k;
bool comp_sa(int i,int j){
    if(rk[i]!=rk[j])return rk[i]<rk[j];
    else{
        int ri = i+k<=n?rk[i+k]:-1;
        int rj = j+k<=n?rk[j+k]:-1;
        return ri<rj;
    }
}
void calc_sa(){
    for(int i=0;i<=n;i++){
        rk[i] = s[i];
        sa[i] = i;
    }
    for(k=1;k<=n;k=k*2){
        sort(sa,sa+n,comp_sa);
        tmp[sa[0]] = 0;
        for(int i=0;i<n;i++)
            tmp[sa[i+1]] = tmp[sa[i]]+(comp_sa(sa[i],sa[i+1])?1:0);
        for(int i=0;i<n;i++){
            rk[i] = tmp[i];
        }
    }
}


//后缀数组2 o(nlogn)比上面的快，基于基数排序算法 hdu1403
#include<bits/stdc++.h>
#define N 200010
using namespace std;
 
char s[N];
int sa[N],t[N],t2[N],c[N],n,rak[N],height[N];
 
void build_sa(int m,char *s)
{
    int i,*x=t,*y=t2;
    for (i=0;i<m;i++)c[i]=0;
    for (i=0;i<n;i++)c[x[i]=s[i]]++;
    for (i=1;i<m;i++)c[i]+=c[i-1];
    for (i=n-1;i>=0;i--)sa[--c[x[i]]]=i;
    for (int k=1;k<=n;k<<=1)
    {
        int p=0;
        for (i=n-k;i<n;i++)y[p++]=i;
        for (i=0;i<n;i++)if (sa[i]>=k)y[p++]=sa[i]-k;
        for (i=0;i<m;i++)c[i]=0;
        for (i=0;i<n;i++)c[x[y[i]]]++;
        for (i=0;i<m;i++)c[i]+=c[i-1];
        for (i=n-1;i>=0;i--) sa[--c[x[y[i]]]]=y[i];
        swap(x,y);
        p=1; x[sa[0]]=0;
        for (i=1;i<n;i++)
            x[sa[i]]=y[sa[i-1]]==y[sa[i]]&&y[sa[i-1]+k]==y[sa[i]+k]?p-1:p++;
        if (p>=n) break;
        m=p;
    }
}
 
void getheight()
{
    int i,j,k=0;
    for (i=0;i<n;i++)rak[sa[i]]=i;
    for (i=0;i<n;i++)
    {
        if (k)k--;
        if (!rak[i])continue;
        j=sa[rak[i]-1];
        while (s[i+k]==s[j+k])k++;
        height[rak[i]]=k;
    }
}
 
int main()
{
    while (~scanf("%s",s))
    {
        n=strlen(s);
        s[n]='0';   int nn=n,ans=0;
        scanf("%s",s+n+1);
        n=strlen(s);
        build_sa(200,s);
        getheight();
        for (int i=1;i<n;i++)
        {
            int x=sa[i-1],y=sa[i];
            if (x>y) swap(x,y);
            if (x<nn && nn<y) ans=max(ans,height[i]);
        }
        printf("%d\n",ans);
    }
    return 0;
}

    
//字符串哈希，简单
unsigned int BKDRHash(char *str){
    unsigned int seed = 31,key = 0,
    while(*str)key = key*seed+(*str++);
    return key&0x7fffffff;
}


//字符串哈希+预处理
const int maxn = 1000005;
unsigned long long H[maxn];
unsigned long long xp[maxn];
char s[maxn];
const int x = 131;//随便取个素数

int n,m,pos;

unsigned long long hash[maxn];
//子串i....j的哈希值
unsigned long long gethash(int i,int j)
{
    int L = j-i+1;
    return H[i] - H[i+L]*xp[L];
}
void init()
{
    H[n] = 0;
    for(int i=n-1;i>=0;i--)H[i] = H[i+1]*x + (s[i]-'a');
    xp[0] = 1;
    for(int i=1;i<=n;i++)xp[i] = xp[i-1]*x;
}


//Treap模板,已测试
inline int read() {
    int x = 0, fh = 1; char ch = getchar();
    for (; !isdigit(ch); ch = getchar()) if (ch == '-') fh = -1;
    for (; isdigit(ch); ch = getchar()) x = (x * 10) + (ch ^ 48);
    return x * fh;
}

const int maxnode = 2e6 + 1e3, inf = 0x7f7f7f7f;
#define ls(o) ch[o][0]
#define rs(o) ch[o][1]
struct Treap {
    int ch[maxnode][2], val[maxnode], prio[maxnode], cnt[maxnode], cnt_sum[maxnode];

    inline void push_up(int o) {
        cnt_sum[o] = cnt[o] + cnt_sum[ls(o)] + cnt_sum[rs(o)];
    }

    void rotate(int &u, int d) {
        int v = ch[u][d];
        ch[u][d] = ch[v][d ^ 1]; ch[v][d ^ 1] = u;
        push_up(u); push_up(v); u = v;
    }

    inline int New(int val_) {
        static int Size = 0, o; o = ++ Size;
        ls(o) = rs(o) = 0; val[o] = val_;
        cnt_sum[o] = cnt[o] = 1;
        prio[o] = rand(); return o;
    }

    void Insert(int &o, int val_) {
        if (!o) { o = New(val_) ; return ; }
        if (val[o] == val_) ++ cnt[o];
        else {
            int d = (val_ > val[o]);
            Insert(ch[o][d], val_);
            if (prio[ch[o][d]] > prio[o]) rotate(o, d);
        }
        push_up(o);
    }

    void Erase(int &o) {
        if (!ls(o) && !rs(o)) { o = 0; return ; }
        int d = (prio[rs(o)] > prio[ls(o)]);
        rotate(o, d); Erase(ch[o][d ^ 1]); push_up(o);
    }

    void Delete(int &o, int val_) {
        if (val[o] == val_) { if (!(-- cnt[o])) Erase(o); push_up(o); return ; }
        int d = (val_ > val[o]);
        Delete(ch[o][d], val_); push_up(o);
    }

    int Rank(int o, int val_) {
        if (val[o] == val_) return cnt_sum[ls(o)];
        int d = (val_ > val[o]);
        return d * (cnt_sum[ls(o)] + cnt[o]) + Rank(ch[o][d], val_);
    }

    int Kth(int o, int k) {
        int res = k - cnt_sum[ls(o)];
        if (res <= 0) return Kth(ls(o), k);
        if (res > cnt[o]) return Kth(rs(o), res - cnt[o]);
        return val[o];
    }

    int Pre(int o, int val_) {
        int res = -inf, d = (val[o] < val_);
        if (!o) return res; if (d) res = val[o];
        return max(res, Pre(ch[o][d], val_));
    }

    int Suf(int o, int val_) {
        int res = inf, d = (val[o] > val_);
        if (!o) return res; if (d) res = val[o];
        return min(res, Suf(ch[o][val_ >= val[o]], val_));
    }
} T;

//val必须存在treap中才能调用T.Rank
//必须存在第val_大才能调用T.Kth
int rt = 0;
int main () {
    srand(time(0));
    int n = read();
    while (n --) {
        int opt = read(), val_ = read();
        if (opt == 1) T.Insert(rt, val_);
        if (opt == 2) T.Delete(rt, val_);
        if (opt == 3) printf ("%d\n", T.Rank(rt, val_) + 1);
        if (opt == 4) printf ("%d\n", T.Kth(rt, val_) );
        if (opt == 5) printf ("%d\n", T.Pre(rt, val_) );
        if (opt == 6) printf ("%d\n", T.Suf(rt, val_) );
    }
    return 0;
}

//整数输入挂
inline int read() {
    int x = 0, fh = 1; char ch = getchar();
    for (; !isdigit(ch); ch = getchar()) if (ch == '-') fh = -1;
    for (; isdigit(ch); ch = getchar()) x = (x * 10) + (ch ^ 48);
    return x * fh;
}


//单调栈
int n;
int h[100005];
int L[100005];
int R[100005];
int st[100005];
void getLR()
{
    int pos=0;
    L[1]=1;
    st[++pos] = 1;
    for(int i=2;i<=n;i++){
        while(pos>=1&&h[st[pos]]>=h[i])pos--;
        L[i]=pos==0?1:st[pos]+1;
        st[++pos]=i;

    }
    pos=0;
    R[n]=n;
    st[++pos] = n;
    for(int i=n-1;i>=1;i--){
        while(pos>=1&&h[st[pos]]>=h[i])pos--;
        R[i]=pos==0?n:st[pos]-1;
        st[++pos]=i;
    }
}

















