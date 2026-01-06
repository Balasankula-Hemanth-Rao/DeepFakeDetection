import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Legend } from 'recharts';
import { motion } from 'framer-motion';
import { TrendingUp, ShieldCheck, AlertTriangle, Activity } from 'lucide-react';

interface AnalysisRecord {
  id: string;
  upload_timestamp: string;
  result?: {
    prediction: string;
    confidence_score: number;
  };
}

interface AnalyticsChartsProps {
  analyses: AnalysisRecord[];
}

export const AnalyticsCharts = ({ analyses }: AnalyticsChartsProps) => {
  const realCount = analyses.filter(a => a.result?.prediction === 'REAL').length;
  const fakeCount = analyses.filter(a => a.result?.prediction === 'FAKE').length;
  const pendingCount = analyses.filter(a => !a.result).length;
  
  const pieData = [
    { name: 'Authentic', value: realCount, color: 'hsl(var(--success))' },
    { name: 'Deepfake', value: fakeCount, color: 'hsl(var(--destructive))' },
    { name: 'Pending', value: pendingCount, color: 'hsl(var(--muted-foreground))' },
  ].filter(d => d.value > 0);

  // Group analyses by week
  const getWeeklyData = () => {
    const weeks: { [key: string]: { real: number; fake: number } } = {};
    
    analyses.forEach(analysis => {
      if (!analysis.result) return;
      
      const date = new Date(analysis.upload_timestamp);
      const weekStart = new Date(date);
      weekStart.setDate(date.getDate() - date.getDay());
      const weekKey = weekStart.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
      
      if (!weeks[weekKey]) {
        weeks[weekKey] = { real: 0, fake: 0 };
      }
      
      if (analysis.result.prediction === 'REAL') {
        weeks[weekKey].real++;
      } else {
        weeks[weekKey].fake++;
      }
    });

    return Object.entries(weeks)
      .slice(-6)
      .map(([week, data]) => ({
        week,
        ...data
      }));
  };

  const weeklyData = getWeeklyData();
  
  // Calculate average confidence
  const avgConfidence = analyses.reduce((acc, a) => {
    if (a.result) return acc + a.result.confidence_score;
    return acc;
  }, 0) / (analyses.filter(a => a.result).length || 1);

  const stats = [
    { 
      label: 'Total Scans', 
      value: analyses.length, 
      icon: Activity,
      color: 'text-primary'
    },
    { 
      label: 'Authentic', 
      value: realCount, 
      icon: ShieldCheck,
      color: 'text-success'
    },
    { 
      label: 'Deepfakes', 
      value: fakeCount, 
      icon: AlertTriangle,
      color: 'text-destructive'
    },
    { 
      label: 'Avg Confidence', 
      value: `${Math.round(avgConfidence * 100)}%`, 
      icon: TrendingUp,
      color: 'text-primary'
    },
  ];

  if (analyses.length === 0) {
    return null;
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="space-y-6"
    >
      {/* Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {stats.map((stat, index) => (
          <motion.div
            key={stat.label}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: index * 0.1 }}
          >
            <Card className="glass">
              <CardContent className="p-4 flex items-center gap-3">
                <div className={`p-2 rounded-lg bg-muted ${stat.color}`}>
                  <stat.icon className="w-5 h-5" />
                </div>
                <div>
                  <p className="text-2xl font-bold">{stat.value}</p>
                  <p className="text-xs text-muted-foreground">{stat.label}</p>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </div>

      {/* Charts Row */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Pie Chart */}
        <Card className="glass">
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Detection Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={50}
                    outerRadius={70}
                    paddingAngle={5}
                    dataKey="value"
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    labelLine={false}
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: 'hsl(var(--card))',
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '8px'
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Bar Chart - Weekly Activity */}
        <Card className="glass">
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Weekly Activity</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-48">
              {weeklyData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={weeklyData}>
                    <XAxis 
                      dataKey="week" 
                      tick={{ fontSize: 10 }}
                      stroke="hsl(var(--muted-foreground))"
                    />
                    <YAxis 
                      tick={{ fontSize: 10 }}
                      stroke="hsl(var(--muted-foreground))"
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: 'hsl(var(--card))',
                        border: '1px solid hsl(var(--border))',
                        borderRadius: '8px'
                      }}
                    />
                    <Legend />
                    <Bar dataKey="real" name="Authentic" fill="hsl(var(--success))" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="fake" name="Deepfake" fill="hsl(var(--destructive))" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-full flex items-center justify-center text-muted-foreground text-sm">
                  Not enough data for weekly chart
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </motion.div>
  );
};
